---

layout: post
tags: [LLM, NLP, AI Infra]
title: Inside vLLM and KV Cache
date: 2025-9-15
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

随着大语言模型(LLM)在各个领域的广泛应用，如何高效地部署和推理这些模型成为了一个关键挑战。传统的模型推理服务往往面临着内存利用率低、吞吐量受限、延迟不可控等问题，这些瓶颈严重制约了LLM在生产环境中的规模化应用。vLLM作为一个专为LLM优化的高性能推理服务框架，通过一系列创新的技术方案，有效解决了这些痛点问题。

本文将深入剖析vLLM的核心架构和关键技术实现，从底层的内存管理机制到上层的服务调度策略，全面解析其如何实现高效的LLM推理服务。我们将重点探讨以下几个核心技术模块：

**PagedAttention机制**：借鉴操作系统中虚拟内存管理的思想，vLLM提出了PagedAttention技术，将KV Cache按页进行管理，实现了内存的按需分配和高效利用。这种设计不仅显著降低了内存碎片化问题，还支持了动态序列长度处理，使得内存利用率相比传统方案提升了数倍。

**Continuous Batching(连续批处理)**：传统的静态批处理方式存在严重的计算资源浪费问题，特别是当批内序列长度差异较大时。vLLM的连续批处理技术支持序列的动态加入和完成，实现了真正的流水线式处理，大幅提升了系统吞吐量和资源利用效率。

**Prefix Caching(前缀缓存)**：在实际应用中，很多请求往往共享相同的前缀内容（如系统提示词、模板等）。vLLM通过智能的前缀缓存机制，能够复用已计算的KV Cache，避免重复计算，显著降低了推理延迟和计算开销。

**Speculative Decoding(推测解码)**：为了进一步提升生成速度，vLLM集成了推测解码技术，通过使用较小的draft模型预先生成候选token，然后由主模型进行验证，实现了在保证输出质量的前提下大幅加速文本生成过程。

**分布式架构与多GPU协同**：面对大模型参数量不断增长的趋势，vLLM提供了完善的分布式解决方案，支持张量并行、流水线并行等多种并行策略，能够在多GPU、多节点环境下实现高效的模型推理，满足大规模生产环境的性能需求。

**动态扩缩容与服务化**：作为一个面向生产的推理框架，vLLM不仅关注性能优化，还提供了完整的服务化能力，包括请求路由、负载均衡、自动扩缩容等功能，使得用户能够轻松构建高可用、高性能的LLM服务集群。

通过对这些关键技术的深入分析，我们将展现vLLM如何通过系统性的优化设计，在保证推理质量的前提下，实现了相比传统方案数倍甚至数十倍的性能提升。这些技术创新不仅推动了LLM推理服务的发展，也为整个AI基础设施领域提供了宝贵的设计思路和实践经验。一共分为五个部分：

• **LLM engine**以及**engine core**：包含了vLLM的基础架构（调度、PagedAttention、continous batching）
• **Advanced Features 高级特性**：chunked prefill(分块预填充)、prefix caching(前缀缓存)、guided&speculative decoding(引导预测编码)、disaggregated P/D(Prefill-decoding分离)
• **Scaling Up**：单进程执行到多进程多GPU
• **Server Layer**：分布式集群服务化部署
• **Benchmarks**与**Auto-tuning**：平衡延迟和吞吐

# LLM Engine & Engine Core

在vLLM中，LLM Engine是最基础的block，在离线场景中，它本身就支持高吞土地推理。以下是一个简单的离线推理例子：

```Python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()

## Environment vars:
##   VLLM_USE_V1="1" # we're using engine V1
##   VLLM_ENABLE_V1_MULTIPROCESSING="0" # we're running in a single process
## 
```

我们调用模型执行器的 `execute_model`，它会委派给 `Worker`，而 `Worker` 又会继续委派给 `model runner`。

主要步骤如下：

- **更新状态** —— 从 `input_batch` 中裁剪已完成的请求；更新与前向传播相关的其他metadata（例如每个请求的 KV cache 块数，用于在分页的 KV cache 内存中建立索引）。
- **准备输入** —— 将缓冲区从 `CPU→GPU` 复制；计算位置；构建 `slot_mapping`（示例中会详细说明）；构造注意力metadata。
- **前向传播** —— 使用自定义的 PagedAttention 内核运行模型。所有序列会被展平并连接为一个长的“超级序列”。位置索引与注意力掩码确保每个序列只关注自己的 token，从而在不使用右侧填充的情况下实现持续批处理。
- **收集最后一个 token 的状态** —— 为每个序列的最终位置提取隐藏状态并计算 `logits`。
- **采样** —— 按照采样配置（贪心、温度、`top-p`、`top-k` 等）从计算出的 `logits` 中采样 token。

前向步骤本身有两种执行模式：

- **Eager 模式** —— 在启用 eager 执行时运行标准的 PyTorch 前向传播。
- **“捕获”模式** —— 在未强制启用 eager 的情况下，执行或回放预先捕获的 CUDA Graph（还记得在引擎构建的初始化 KV cache 过程中我们已经捕获了它们）。

这些配置有：

- 离线模式（无Web服务或分布式系统架构）；
- 同步执行（所有执行都在单个阻塞进程中进行）；
- 单GPU（无数据/模型/流水线/专家并行；DP/TP/PP/EP = 1）；
- 使用标准transformer结构（支持像Jamba这样的混合模型需要更复杂的混合KV缓存内存分配器）。

在这个例子中，我们做了两件事：
    1. 实例化了一个engine；
    2. 通过给定的prompt来调用 `generate` 方法去做采样。

## LLM Engine constructor

对于engine而言，核心的组成部分有：

  - vLLM config：包含模型配置的全部信息、cache、并行策略等；
  - processer：通过validation、tokenization和processing将 `raw input` -> `EngineCoreRequests`;
  - engine core client：在我们的例子中使用了 `InprocClient` ，基本上等于 `EngineCore` ，会逐步搭建成 `DPLBAsyncMPClient` ，允许大规模提供服务；
  - output processor：将 `raw EngineCoreOutputs` -> `RequestOutputs` 转换给用户看。

至于 `EngineCore` 本身由以下组件组成：

- 模型执行器 (Model Executor): 驱动模型的前向传播。我们目前接触的是在单个GPU上使用单个Worker进程的 `UniProcExecutor`，后续会逐步扩展到支持多GPU的 `MultiProcExecutor`。
- 结构化输出管理器 (Structured Output Manager): 用于引导式解码（稍后会详细介绍）。
- 调度器 (Scheduler): 决定哪些请求进入下一个引擎步骤，它进一步包含：
    - 策略设置 (policy setting): 可以是FCFS（先到先得）或优先级（高优先级请求优先处理）。
    - 等待和运行队列 (waiting and running queues)。
    - KV缓存管理器 (KV cache manager): PagedAttention机制的核心。

KV Cache Manager 维护了 `free_block_queue`，也就是可用的 KV Cache blocks组成的资源池；规模往往能到几十万，取决于显存与块大小。当 PagedAttention 执行时，这些块承担索引作用，将各个 token 对应到它们的 KV Cache block。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-1.png)


```
其中对于一个标准transformer层（非MLA）的block size可以通过以下方式计算：

2 (key/value) * block_size (default=16) * num_kv_heads * head_size * dtype_num_bytes (e.g. 2 for bf16)
```

当model excutor构建时，会创建一个 `Worker` 对象，并执行三个主要步骤（在使用 `MultiProcExecutor` 时，这些步骤会在不同 GPU 上的每个 worker 进程中独立运行）：

- 初始化设备:
    - 为该 worker 分配 CUDA 设备（e.g. `cuda:0` ），并检查模型的 dtype 是否受支持（e.g. `bf16` ）
    - 根据设定的 `gpu_memory_utilization` （e.g. 0.8 → 80% of total VRAM）验证显存是否充足
    - 配置分布式设置（ DP / TP / PP / EP, etc.）
    - 实例化 `model_runner` （包含采样器、KV cache，以及forward pass的buffers如 `input_ids` 、 `positions`, etc.）
    - 实例化 `InputBatch` 对象（包含 CPU-side forward pass buffering、KV cache indexing、sampling metadata等）

- 加载模型:
    - 实例化模型架构
    - 加载模型权重
    - 调用 `model.eval()` （PyTorch 的推理模式）
    - 可选：对模型调用 `torch.compile()`

- 初始化 KV Cache:
    - 获取按层的 KV cache spec。通常为 `FullAttentionSpec` （同质 Transformer），但在引入混合模型（滑动窗口、Transformer/SSM，如 Jamba）后变得更复杂
    - 运行一次dummy/profiling forward pass，并记录 GPU 内存快照，用于计算在可用显存中能容纳多少 KV cache blocks
    - 为注意力层分配、reshape并绑定 KV cache tensors
    - 准备 `attention metadata`（如将后端设置为 `FlashAttention` ），供后续前向过程中的内核使用
    - 若未提供 `--enforce-eager` ，则针对若干预热批大小进行空跑并捕获 CUDA graph。CUDA graph会把整段 GPU 工作记录为一个 DAG；之后在前向过程中，我们会启动/回放这些预先捕获（预烘焙）的 CUDA graph，削减 kernel 启动开销，因而时延更低。

我们在这里抽象掉了许多底层细节，但以上是后文将反复引用的核心组件与流程。引擎初始化完成后，继续进入 `generate` 函数。

## Generate function

第一步是对请求进行校验并送入 engine 。对于每个 prompt，我们会：

1. 创建一个唯一的请求 ID，并记录其到达时间。
2. 调用输入预处理器对 prompt 进行标记化（tokenize），返回一个字典 dictionary，包含 `prompt` 、 `prompt_token_ids` ，以及一个 `type`（如 text、tokens、embeds, etc.）。
3. 将这些信息打包成一个 `EngineCoreRequest` ，并添加优先级、采样参数及其他metadata。
4. 将请求传入 engine core，core 会将其包装为一个 `Request` 对象并将状态设为 `WAITING` ；随后把该请求加入调度器的等待队列（若为先来先服务 FCFS 则使用 append；若为优先级调度则使用 heap-push）。

至此，引擎已经“进料”，执行即可开始。在同步引擎示例中，只会处理这些初始 prompt——运行过程中无法插入新请求。相反，异步引擎支持在运行中注入请求（即“持续批处理” continuous batching）：在每一步之后，同时考虑新请求与已有请求。

```
前向传播将 batch 扁平化为单序列，配合高效的定制 kernel 处理路径，使得即使在同步引擎中也天然具备 continuous batching 能力。
```

接下来，只要仍有请求待处理，引擎就会反复调用 `step()` 函数。每一步包含三个阶段：
- 调度（Schedule）：选择本步要运行的请求（ decode ，and/or (chunked) prefill ）。
- 前向传播（Forward pass）：运行模型并进行 token 采样。
- 后处理（Postprocess）：将采样得到的 token ID 追加到各个 `Request` ，执行反标记化（`detokenize`），并检查停止条件。若某个请求已完成，则进行清理（例如把它的 KV Cache block 归还到 `free_block_queue` ），并提前返回该请求的输出。

📝 停止条件包括：
- 请求超过长度上限（ `max_model_length` 或其自身的 `max_tokens` ）。
- 采样到 EOS ID（除非启用了 `ignore_eos` → 在 benchmarking 中可用于强制生成固定数量的输出 token）。
- 采样到的 token 匹配到采样参数中指定的任意 `stop_token_ids` 。
- 输出中出现停止字符串（stop strings）——我们会将输出截断到首次出现停止字符串的位置，并在引擎中终止该请求（注意： stop_token_ids 会保留在输出中，而停止字符串不会保留）。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-2.png)

在流式模式中，我们会在生成过程中实时发送中间 token，但这里暂不展开。接下来，我们将更详细地讨论调度。

## Scheduler

推理引擎处理两种主要类型的工作负载：

- **Prefill 请求** ： 对所有 prompt token 进行前向传播。这些通常是计算密集型的（阈值取决于硬件和prompt长度）。最后，我们从最终 token 位置的概率分布中采样一个 token。
- **Decode 请求** ： 仅对最新的 token 进行前向传播。所有较早的 KV 向量已经被缓存。这些是 `memory-bandwidth-bound` 的，因为我们仍然需要加载所有 LLM 权重（和 KV cache）来计算一个 token。

V1 scheduler 可以在同一步骤中混合处理两种类型的请求，这得益于更智能的设计选择。相比之下，V0 engine 一次只能处理 prefill 或 decode 中的一种 workload。

Scheduler 优先处理 decode 请求——即那些已经在运行队列中的请求。对于每个这样的请求，它会：
1. 计算要生成的新 token 数量（由于推测解码和异步调度，不总是会在第一步做这些事情，——稍后会详细介绍）。
2. 调用 KV cache manager 的 `allocate_slots` 函数（详细信息见下文）。
3. 更新 token budget：不断减少第 1 步计算得到的 token 数量。

之后，它处理等待队列中的 prefill 请求：
1. 检索已计算块的数量（如果禁用前缀缓存则返回 0——稍后会介绍）。
2. 调用 KV cache manager 的 `allocate_slots` 函数。
3. 将请求从等待队列中弹出并移动到运行队列，将其状态设置为 `RUNNING`。
4. 更新 token budget。

现在让我们看看 `allocate_slots` 的作用：
1. **计算块数量** — 确定必须分配多少个新的 KV cache 块（n）。每个块默认存储 16 个 token。例如，如果一个 prefill 请求有 17 个新 token，我们需要 ceil(17/16) = 2 个块。
2. **检查可用性** — 如果管理器池中没有足够的块，则提前退出。根据是 decode 还是 prefill 请求，引擎可能会尝试重计算抢占（V0 中支持交换抢占），通过驱逐低优先级请求（调用 `kv_cache_manager.free` 将 KV 块返回到块池），或者可能跳过调度并继续执行。
3. **分配块** — 通过 KV cache manager 的协调器，从块池（前面提到的 `free_block_queue` 双向链表）中获取前 n 个块。存储到 `req_to_blocks`，这是将每个 `request_id` 映射到其 KV cache block list的字典。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-3.png)

最终，我们准备好做一次前向传递了。

## Run Forward pass

我们调用模型执行器的 `execute_model`，它会委派给 `Worker`，而 `Worker` 又进一步委派给 `model runner`。

主要步骤如下：

- **更新状态** —— 从 `input_batch` 中裁剪已完成的请求；更新与前向传播相关的其他metadata（例如每个请求的 KV cache 块数，用于在分页的 KV cache 内存中建立索引）。
- **准备输入** —— 将缓冲区从 `CPU→GPU` 复制；计算位置；构建 `slot_mapping`（示例中会详细说明）；构造注意力metadata。
- **前向传播** —— 使用自定义的 PagedAttention 内核运行模型。所有序列会被展平并拼接为一个长的“超级序列”。位置索引与注意力掩码确保每个序列只关注自身的 token，从而在不进行右侧填充的情况下实现 continuous batching。
- **收集最后一个 token 的状态** —— 为每个序列的最终位置提取隐藏状态并计算 `logits`。
- **采样** —— 按照采样配置（greedy、temperature、top-p、top-k 等）从计算得到的 `logits` 中采样 token。

前向步骤本身有两种执行模式：

- **Eager Mode* —— 启用 eager 执行时运行标准的 PyTorch 前向传播。
- **“Capture” Mode** —— 在未强制启用 eager 的情况下，执行/回放预先捕获的 CUDA Graph（还记得我们在引擎构建的初始化 KV cache 过程中已捕获这些 graph）。

下面是一个具体示例，可帮助你更清晰地理解 continuous batching 和 PagedAttention：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-4.png)

# Advanced Features — extending the core engine logic

在掌握基本的引擎流程后，我们可以继续了解一些高级特性。

我们已经讨论了抢占（preemption）、PagedAttention 和 continuous batching。

接下来，我们将深入讲解：

- Chunked prefill
- Prefix caching
- Guided decoding
- Speculative decoding
- Disaggregated P/D

## Chunked prefill

Chunked prefill（分块式 prefill）是一种通过将长 prompt 的 prefill 步骤拆分为更小的 chunk 来处理长 prompt 的技术。若不使用该方法，一个非常长的请求可能会在某次 `engine step` 中长时间独占执行，阻止其他 prefill 请求运行，从而推迟所有其他请求并显著提高它们的延迟。

例如，令每个 chunk 包含 n (=8) 个 token，并用小写字母以 “-” 分隔来标记。一个长提示 `P` 可以表示为 `x-y-z`，其中 `z` 是未完成的 chunk（例如仅包含 2 个 tokens）。执行 `P` 的完整 prefill 至少需要 ≥ 3 个 `engine step`（如果某一步未被调度执行，还可能需要更多），并且只有在最后一个分块 prefill 步骤中我们才会采样一个新 token。

以下是同一示例的可视化说明：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-5.png)

实现很直接：为每个 engine step 设定“新增 token 数量”的上限。当请求的数量超过 `long_prefill_token_threshold` 时，将其重置为该阈值。其余流程由底层的索引逻辑（前文已述）自动处理。

在 vLLM V1 中，通过将 `long_prefill_token_threshold` 设置为正整数即可启用 chunked prefill。（从技术上讲，即使未显式设置也可能发生：若 prompt 长度超过 token 预算，我们会先截断它，并以分块 prefill 的方式运行。）

## Prefix Caching

为了解释 prefix caching 的工作原理，可以参考以下代码：

```python
from vllm import LLM, SamplingParams

long_prefix = "<a piece of text that is encoded into more than block_size tokens>"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(long_prefix + prompts[0], sampling_params)
    outputs = llm.generate(long_prefix + prompts[1], sampling_params)

if __name__ == "__main__":
    main()
```
Prefix caching 用于避免对多个 prompt 共享的开头部分重复计算（因此称为 **“前缀 Prefix”** ）。

关键在于 `long_prefix`：它被定义为长度超过一个 KV cache block 的前缀（默认每块 16 tokens）。为简化示例，假设 `long_prefix` 的长度恰好为 `n × block_size`（其中 `n ≥ 1`）。

也就是说，它必须与块边界完全对齐——否则我们必须重新计算 `long_prefix_len % block_size` 个 tokens，因为不完整的块无法被缓存。若不使用 prefix caching，每次处理一个具有相同 `long_prefix` 的新请求时，都要重新计算这 `n × block_size` 个 tokens。

而使用 prefix caching 时，这些 tokens 只需计算一次（其 KV 存入分页的 `KV cache` 内存）并被复用，因此仅需处理新的 prompt tokens。这会显著加速 prefill 请求（但对 decode 无帮助）。

那么在 vLLM 中如何工作？

在首次 `generate` 调用的调度阶段，`kv_cache_manager.get_computed_blocks` 内，engine 会调用 `hash_request_tokens`：

- 将 `long_prefix + prompts[0]` 按 16-token 切分为 chunks。
 - 对每个完整 chunk 计算一个 hash（使用内建 `hash` 或 `SHA-256`，后者更慢但 hash 冲突更少）。该 hash 组合了上一块的 hash、当前 tokens 以及可选metadata。可选metadata包括：`MM hash`、`LoRA ID`、`cache salt`（注入首块的 hash，保证只有携带该 `cache salt` 的请求能复用这些块）。
- 每个结果以 `BlockHash` 对象存储，包含其 hash 与 token IDs；函数返回一个 block hashes 列表。

该列表写入 `self.req_to_block_hashes[request_id]`。

随后，engine 调用 `find_longest_cache_hit`，检查这些 hash 是否已存在于 `cached_block_hash_to_block` 中。对于首个请求，通常不会有命中。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-6.png)


然后我们调用 `allocate_slots`，它会进一步调用 `coordinator.cache_blocks`，将新的 `BlockHash` 条目与已分配的 `KV cache` blocks 关联，并把映射记录到 `cached_block_hash_to_block`。

随后，前向传播会在分页的 `KV cache` 内存中填充对应的 KV，覆盖我们上面分配的这些 `KV cache` blocks。

在经历多个 `engine step` 后，系统会继续分配更多 `KV cache` blocks。但在本示例中这并不重要，因为前缀在 `long_prefix` 之后就立即发生了差异。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-7.png)

第二次以相同前缀调用 `generate` 时，前述步骤 1–3 会再次执行，但此时 `find_longest_cache_hit` 会（通过线性搜索）为全部 `n` 个块找到命中，engine 可直接复用这些 `KV cache` blocks。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-8.png)

如果最初的请求仍在运行，这些块的引用计数（reference count）会增加（例如变为 2）。在本例中，第一个请求已经完成，因此这些块被释放回到池中，其引用计数重置为 0。由于我们能够从 `cached_block_hash_to_block` 取回它们，表明这些块仍然有效（KV cache manager 的逻辑就是这样设计的），所以我们只需再次将它们从 `free_block_queue` 中移除即可复用。

`KV cache` blocks 只有在即将从 `free_block_queue`（从左端弹出）重新分配时才会被判定为“无效”。此时如果发现该块仍有关联的 hash 并存在于 `cached_block_hash_to_block` 中，我们会清除该块的 hash，并将其从 `cached_block_hash_to_block` 中移除，以确保它不能再通过 prefix caching 被复用（至少不能用于旧的前缀）。

这就是 prefix caching 的核心：不要重复计算已经见过的前缀——直接复用它们的 `KV cache`！

如果理解了这个示例，也就理解了 PagedAttention 的工作方式。

Prefix caching 默认启用。若要关闭：将 `enable_prefix_caching = False`。

## Guided Decoding (FSM)

Guided decoding（引导式解码）是一种在每个解码步对 `logits` 施加约束的技术，约束由基于语法的有限状态机（FSM）定义。这确保了只会采样语法允许的 token。

这是一个非常强大的设定：你可以强制执行从正则语法（Chomsky type-3，例如任意正则表达式模式）到上下文无关语法（type-2，覆盖大多数编程语言）的一切约束。

为使其更具体，我们先从最简单的示例入手，并在先前的代码基础上继续构建：

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

prompts = [
    "This sucks",
    "The weather is beautiful",
]

guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

在给出的 toy example 中（假设字符级 tokenization）：在 prefill 阶段，FSM 会对 `logits` 进行掩蔽，使得只有 “P” 或 “N” 可以被采样。若采样到 “P”，FSM 将转入 “Positive” 分支；下一步仅允许 “o”，依此类推。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-9.png)

vLLM 中的工作方式：

1. 在构造 LLM 引擎时，会创建 `StructuredOutputManager`；它可访问 tokenizer，并维护一个 `_grammar_bitmask` 张量；
2. 当添加请求时，状态被设置为 `WAITING_FOR_FSM`，`grammar_init` 会选择后端编译器（例如 `xgrammar` ，这里的大部分复杂度都隐藏在诸如 `xgrammar` 等第三方库中）；
3. 针对该请求的语法会异步编译；
4. 在调度过程中，如果异步编译已完成，状态切换为 `WAITING`，并将 `request_id` 加入 `structured_output_request_ids`；否则将其放入 `skipped_waiting_requests`，在下一次引擎步（engine step）重试。
5. 在调度循环结束后（仍处于调度阶段），如果存在 FSM 请求，`StructuredOutputManager` 会让后端准备/更新 `_grammar_bitmask`。
6. 当前向传播产生 `logits` 后，`xgr_torch_compile` 的函数会将位掩码展开到词表大小（由于使用 32 位整数，展开比例为 32×），并将不允许的 `logits` 置为 `-∞`。
7. 在采样下一个 token 之后，通过 `accept_tokens` 推进该请求的 FSM。直观上我们在 FSM 图上移动到下一个状态。

其中第 6 步值得进一步澄清：

若 `vocab_size = 32`，`_grammar_bitmask` 是一个整数；其二进制表示编码了哪些 token 被允许（“1”）与不允许（“0”）。例如，“101…001”会展开为长度为 32 的数组 `[1, 0, 1, …, 0, 0, 1]`；值为 0 的位置对应的 `logits` 被置为 `-∞`。

对更大的词表，会使用多个 32 位字，并按需展开/拼接。后端（例如 `xgrammar`）负责依据当前 FSM 状态生成这些位模式。

以下还有一个更简单的示例：`vocab_size = 8` 且使用 8 位整数（适合结合可视化来理解）。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-10.png)

在 vLLM 中，你可以通过传入所需的 `guided_decoding` 配置来启用该功能。

## Speculative Decoding

在自回归生成（autoregressive generation）中，每产生一个新 token 都需要对 LLM 做一次前向传播（forward pass）。这个操作的计算开销非常大，因为每一步都要重新加载并应用全部模型权重，只为计算一个 token！（假设 `batch size == 1`，更一般的情况是 `B`）

`Speculative decoding` 通过引入一个更小的 `“Draft LM”` 来加速。Draft LM 以更低成本提出 `k` 个 token 的候选。但我们并不希望最终从这个小模型的结果中直接采样，因为它只是用来猜测可能的续写。我们最终的采样结果仍由大型模型来决定哪些候选是有效的。

具体步骤如下：

1. **Draft** ：使用小模型在当前上下文上运行，提出 `k` 个 token；
2. **Verify** ：使用大模型在“上下文 + k 个草稿 token”上运行一次。这会为这 `k` 个位置外加一个额外位置产生概率（因此得到 `k+1` 个候选）；
3. **Accept/Reject** ：从左到右遍历这 `k` 个草稿 token：
  - 若大模型对该草稿 token 的概率 ≥ 草稿模型的概率，则接受它；
  - 否则，以 `p_large(token) / p_draft(token)` 的概率接受它；
  - 在第一次拒绝处停止，或者接受所有 `k` 个草稿 token；
    - 若所有 `k` 个草稿 token 都被接受，还可以从大模型“免费”采样额外的第 `k+1` 个 token（因为我们已经计算了该分布）。
    - 若发生了拒绝，则在该位置构造一个重新平衡的分布（`p_large - p_draft`，最小值钳制为 0，并归一化为 1），并从中采样最后一个 token。

**Why this works**：尽管我们使用小模型提出候选，但accept/reject规则保证了在期望意义上，序列的分布与逐 token 从大型模型采样的结果完全一致。这意味着 speculative decoding 在统计上等价于标准的自回归解码，但潜在获得更快的decoding速度，因为一次大型模型的前向传播即可产出至多 `k+1` 个 token。

vLLM V1 不支持“LLM draft model”的方法，而是实现了更快但精度较低的提议方案：`n-gram`、`EAGLE` 和 `Medusa`。

三者的一句话概述：
- `n-gram`：取最后 `prompt_lookup_max` 个 token；在序列中寻找此前的匹配；若找到，则提出紧随该匹配之后的 `k` 个 token；否则减小窗口并重试，直到 `prompt_lookup_min`。
- `EAGLE`：对 LLM 做一次 `“model surgery”` ，保留 embeddings 与 LM head，用轻量级 MLP 替换 transformer stack；将其微调为一个廉价draft。
- `Medusa`：在大型模型的顶端（embeddings before LM head）训练 auxiliary linear head，用于并行预测接下来的 `k` 个 token；这些 head 能比单独运行一个小 LM 更高效地提出 token 候选。

在 vLLM 中使用 `ngram` 作为草稿方法来调用 speculative decoding 的方式如下：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config={
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", speculative_config=speculative_config)

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

vLLM 中的工作方式：

**设置阶段（引擎构造期间）：**

1. **初始化设备**：创建一个 `drafter`（草稿模型，例如 `NgramProposer`）和一个 `rejection_sampler`（其部分代码用 Triton 编写）。
2. **加载模型**：加载草稿模型权重（对 n-gram 而言为空操作）。

**在 `generate` 函数中的后续步骤（假设收到全新请求）：**

1. 用大模型运行常规的 prefill 步骤。
2. 前向传播和标准采样后，调用 `propose_draft_token_ids(k)` 从 draft model 采样 `k` 个草稿 token。
3. 将这些存储在 `request.spec_token_ids` 中（更新请求metadata）。
4. 在下一个 engine step 中，当请求处于运行队列时，将 `len(request.spec_token_ids)` 加到"新 token"计数中，以便 `allocate_slots` 为前向传播预留足够的 KV 块。
5. 将 `spec_token_ids` 复制到 `input_batch.token_ids_cpu` 中，形成（上下文 + 草稿）token。
6. 通过 `_calc_spec_decode_metadata` 计算metadata（从 `input_batch.token_ids_cpu` 复制 token，准备 logits 等），然后在草稿 token 上运行大模型前向传播。
7. 不使用常规的 logits 采样，而是用 `rejection_sampler` 从左到右接受/拒绝并产生 `output_token_ids`。
8. 重复步骤 2-7，直到满足停止条件。

理解这一过程的最佳方式是启动调试器并逐步执行代码库，但本节希望能让你对此有所了解。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-11.png)

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-12.png)

## Disaggregated P/D

我此前已经提到过进行 P/D（prefill/decode）解耦的动机。

在实际推理生产过程中，`prefill` 与 `decode` 具有截然不同的性能画像（前者更偏计算受限 compute-bound，后者更偏内存带宽受限 memory-bandwidth-bound），因此将它们的执行拆分是一个合理的设计。这能更紧致地控制延迟——包括 `TFTT`（time-to-first-token）与 `ITL`（inter-token latency），其细节在基准测试章节会进一步展开。

在实践中，我们会运行 `N` 个 vLLM prefill 实例与 `M` 个 vLLM decode 实例，并根据实时请求的混合情况进行自动扩缩。Prefill worker 会将 KV 写入一个专用的 KV-cache 服务；decode worker 则从中读取。这样可以将长且突发的 prefill 与稳定、对延迟敏感的 decode 有效隔离。

那么在 vLLM 中如何实现？为便于说明，下面的示例使用 `SharedStorageConnector`：这是一个用于展示机制细节的调试型 connector 实现。其中`Connector` 是 vLLM 用于在实例之间交换 KV 的抽象。`Connector` 接口目前尚不稳定，短期内计划进行一些改进，这些改动可能包含不兼容的变更。

我们会启动两个 vLLM 实例（`GPU 0` 用于 prefill，`GPU 1` 用于 decode），然后在它们之间传输 KV cache：

```python

import os
import time
from multiprocessing import Event, Process
import multiprocessing as mp

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

def run_prefill(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)
  llm.generate(prompts, sampling_params)

  prefill_done.set()  # notify decode instance that KV cache is ready

  # To keep the prefill node running in case the decode node is not done;
  # otherwise, the script might exit prematurely, causing incomplete decoding.
  try:
      while True:
          time.sleep(1)
  except KeyboardInterrupt:
      print("Script stopped by user.")

def run_decode(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"

  sampling_params = SamplingParams(temperature=0, top_p=0.95)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)

  prefill_done.wait()  # block waiting for KV cache from prefill instance

  # Internally it'll first fetch KV cache before starting the decoding loop
  outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
  prefill_done = Event()
  prefill_process = Process(target=run_prefill, args=(prefill_done,))
  decode_process = Process(target=run_decode, args=(prefill_done,))

  prefill_process.start()
  decode_process.start()

  decode_process.join()
  prefill_process.terminate()
```

vLLM 中的步骤如下：

1. 实例化（引擎构造期间）, connector 会在两个地方被创建：
  - 在 worker 的设备初始化流程中（`init_worker_distributed_environment` 下），以角色 "worker" 创建 connector；
  - 在 scheduler 的构造函数中，以角色 "scheduler" 创建 connector；
2. Cache 查找：当 scheduler 从等待队列处理 prefill 请求（在本地 prefix-cache 检查之后），会调用 connector 的 `get_num_new_matched_tokens`，以检测 KV-cache 服务中是否存在外部缓存的 token。prefill 场景下该值始终为 0；decode 场景下可能命中。结果会在调用 `allocate_slots` 之前加到本地计数中；
3. 状态更新：scheduler 随后调用 `connector.update_state_after_alloc`，记录命中 cache 的请求（对 prefill 而言是 no-op）；
4. metadata构建：在调度末尾，scheduler 调用 `meta = connector.build_connector_meta`：
    - prefill 将所有 `is_store=True` 的请求加入（用于上传 KV）；
    - decode 将所有 `is_store=False` 的请求加入（用于获取 KV）；
5. Context Manager：在前向传播之前，engine 进入一个 KV-connector 的 context manager：
    - 进入时：调用 `kv_connector.start_load_kv`，对 decode 而言，它会从外部服务器加载 KV 并注入到分页内存；对 prefill 而言是 no-op；
    - 退出时：调用 `kv_connector.wait_for_save`，对 prefill 而言，它会阻塞直至 KV 上传到外部服务器；对 decode 而言是 no-op。

下图给出一个可视化示例：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/250715-13.png)

- 对于 `SharedStorageConnector`，所谓的“external server”其实只是本地文件系统。
- 依据配置，KV 也可以按层进行传输（在每个注意力层前/后进行加载/保存）。
- `decode` 仅在其请求的第一步加载一次外部 KV；之后便在本地进行计算与存储。

# From UniprocExecutor to MultiProcExecutor

# Distributed system serving vLLM

## On the headless server node

## On the API server node

# Benchmarks and auto-tuning - latency vs throughput

# Epilogue

# Acknowledgements

A huge thank you to Hyperstack for providing me with H100s for my experiments over the past year!

Thanks to Nick Hill (core vLLM contributor, RedHat), Mark Saroufim (PyTorch), Kyle Krannen (NVIDIA, Dynamo), and Ashish Vaswani for reading pre-release version of this blog post and providing feedback!

References
vLLM https://github.com/vllm-project/vllm
"Attention Is All You Need", https://arxiv.org/abs/1706.03762
"Efficient Memory Management for Large Language Model Serving with PagedAttention", https://arxiv.org/abs/2309.06180
"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", https://arxiv.org/abs/2405.04434
"Jenga: Effective Memory Management for Serving LLM with Heterogeneity", https://arxiv.org/abs/2503.18292
"Orca: A Distributed Serving System for Transformer-Based Generative Models", https://www.usenix.org/conference/osdi22/presentation/yu
"XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models", https://arxiv.org/abs/2411.15100
"Accelerating Large Language Model Decoding with Speculative Sampling", https://arxiv.org/abs/2302.01318
"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", https://arxiv.org/abs/2401.15077
"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", https://arxiv.org/abs/2401.10774
LMCache, https://github.com/LMCache/LMCache

