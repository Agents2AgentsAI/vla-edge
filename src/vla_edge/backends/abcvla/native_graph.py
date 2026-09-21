"""Full TorchDynamo graph -> one CUDA graph of native operators.
No Inductor/AOT decomposition or eager fallback in the replay path.
"""

import json
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten, tree_map


class NativeCudaGraphBackend:
    def __init__(self, model, evidence_dir=None):
        self.static_addresses = {
            t.data_ptr()
            for t in list(model.parameters()) + list(model.buffers())
            if t.is_cuda
        }
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.graphs = []
        self.graph_modules = []

    def __call__(self, graph_module, example_inputs):
        index = len(self.events)
        # Materialize literal tensor constants once, with the exact native
        # constructor. torch.tensor(..., device='cuda') performs a host copy
        # and cannot run inside CUDA stream capture.
        hoisted = []
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in (
                torch.tensor,
                torch.as_tensor,
            ):
                continue
            flat, _ = tree_flatten((node.args, node.kwargs))
            if any(isinstance(value, torch.fx.Node) for value in flat):
                continue
            value = node.target(*node.args, **node.kwargs)
            if not isinstance(value, torch.Tensor):
                continue
            name = f"_native_constant_{len(hoisted)}"
            graph_module.register_buffer(name, value)
            with graph_module.graph.inserting_before(node):
                replacement = graph_module.graph.get_attr(name)
            node.replace_all_uses_with(replacement)
            graph_module.graph.erase_node(node)
            hoisted.append(
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            )
        graph_module.graph.lint()
        graph_module.recompile()
        # Keep hoisted constant storage alive for every future replay.
        self.graph_modules.append(graph_module)
        static = []
        dynamic_indices = []
        for i, value in enumerate(example_inputs):
            if not isinstance(value, torch.Tensor):
                static.append(value)
            elif not value.is_cuda:
                raise RuntimeError(f"Non-CUDA tensor entered numeric graph: input {i}")
            elif value.data_ptr() in self.static_addresses:
                static.append(value)
            else:
                static.append(value.detach().clone(memory_format=torch.preserve_format))
                dynamic_indices.append(i)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):
                warm = graph_module(*static)
        del warm
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        if self.evidence_dir is not None:
            graph.enable_debug_mode()
        with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
            outputs = graph_module(*static)
        stream.synchronize()
        if self.evidence_dir is not None:
            graph.debug_dump(str(self.evidence_dir / f"cuda_graph_{index}.dot"))
        event = {
            "index": index,
            "fx_nodes": len(list(graph_module.graph.nodes)),
            "graph_breaks": 0,
            "cuda_graph_captured": True,
            "static_weight_and_buffer_inputs": len(static) - len(dynamic_indices),
            "dynamic_tensor_inputs": dynamic_indices,
            "native_operators_preserved": True,
            "literal_tensor_constants_hoisted": hoisted,
        }
        self.events.append(event)
        self.graphs.append(graph)
        if self.evidence_dir is not None:
            (self.evidence_dir / "captures.json").write_text(
                json.dumps(self.events, indent=2)
            )
        print("NATIVE_FULL_CUDA_GRAPH", json.dumps(event), flush=True)

        def run(*inputs):
            for i in dynamic_indices:
                static[i].copy_(inputs[i])
            graph.replay()
            # Native eager returns independent output storage. Do not expose
            # replay buffers that would be overwritten by the next request.
            return tree_map(
                lambda value: (
                    value.clone() if isinstance(value, torch.Tensor) else value
                ),
                outputs,
            )

        return run
