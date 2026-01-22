# 实验感悟

mlhw6整个实现所做的事情就是在测试一个目前学术界发现的现象： **模型在微调后会遗忘原来已有的能力，且在某方向上微调的能力越强，在其他方向上遗忘原来能力的现象越大**

目前有好方法缓解这种情况：

1. 在微调数据中添加一些原来训练模型时的数据
2. 微调数据是由模型(而非人类)生成的能够有效减缓遗忘问题 

实验用数学数据集GSM8K来微调一个模型，然后用安全数据集AILuminate来验证模型的遗忘程度

# 如何解决OutOfMemoryError

我正在微调模型，我的模型加载和微调模型代码如下：

```
model_path = "/home/yxlin/huggingface/Qwen2.5-7B-Instruct"

sft_model = AutoModelForCausalLM.from_pretrained( # Loads the pre-trained model

    pretrained_model_name_or_path=model_path,

)

sft_tokenizer = AutoTokenizer.from_pretrained( # Loads the tokenizer for the model

    pretrained_model_name_or_path=model_path,

)

peft_config = LoraConfig(

    r=16,

    lora_alpha=16,

    lora_dropout=0.05,

    bias='none',

    task_type='CAUSAL_LM',

    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']

)

peft_model = get_peft_model(sft_model, peft_config)



# trainer

training_arguments = SFTConfig( # Configuration for the SFT trainer

    seed=1126,

    data_seed=1126,

    output_dir=f"sft",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    optim="paged_adamw_32bit",

    num_train_epochs=2, # TODO: If you use fixed few-shot examples, increase epoch

    logging_strategy="steps",

    logging_steps=0.1,

    save_strategy="steps",

    save_steps=0.1,

    lr_scheduler_type='linear',

    learning_rate=2e-4, # TODO: Decrease learning rate

    # TODO: Add weight decay

    bf16=True,

    group_by_length=True,

    dataset_text_field='text',

    report_to='none',

)

trainer = SFTTrainer( # Creates the SFT trainer

    model=peft_model,

    train_dataset=formatted_gsm8k,

    peft_config=peft_config,

    processing_class=sft_tokenizer,

    args=training_arguments,

)

trainer.train() # Starts the training process
```

但是报错OutOfMemoryError。


## Solve

要解决 `OutOfMemoryError` 并利用你的两张 GPU，主要有两种策略。具体选择哪种取决于你的显存大小（例如你是 2x24GB 还是 2x16GB）以及你想要的训练方式。

这里有三个层面的解决方案，建议优先尝试 **方案一** 或 **方案二**。

### 核心问题分析

你遇到 OOM 且只用到一张卡的原因是：

1. **默认行为**：HuggingFace 的 `from_pretrained` 默认只会把模型加载到第一张可见的 GPU（cuda:0）或者 CPU 上，除非你显式指定。
2. **显存瓶颈**：7B 模型使用 BF16 精度加载大约需要 14GB 显存，加上优化器状态（Optimizer States）和梯度，显存占用会翻倍。如果你单张卡的显存不足（例如只有 16GB 或 24GB），很容易 OOM。

---

### 方案一：使用 `device_map="auto"`（最简单的代码修改）

这是最直接的修改方式。通过设置 `device_map="auto"`，HuggingFace 会自动计算如何将模型“切分”并放置在两张 GPU 上（例如前 15 层在 GPU0，后 17 层在 GPU1）。这能有效利用两张卡的**总显存**。

**修改后的代码如下：**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

model_path = "/home/yxlin/huggingface/Qwen2.5-7B-Instruct"

# 1. 配置 device_map="auto" 让模型自动分布在两张卡上
sft_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    device_map="auto",  # 关键修改：自动分配层到不同的 GPU
    torch_dtype=torch.bfloat16, # 显式指定精度，防止加载为 fp32 导致显存爆炸
    trust_remote_code=True
)

sft_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# 注意：在使用 device_map 时，TRL 的 SFTTrainer 最好直接传入 base model 和 peft_config
# 让 Trainer 内部处理 PEFT 的包装，这样兼容性更好。
# 如果你坚持手动 get_peft_model，也可以，但 Trainer 可能会报 device 相关的 warning。

training_arguments = SFTConfig(
    seed=1126,
    data_seed=1126,
    output_dir="sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    num_train_epochs=2,
    logging_strategy="steps",
    logging_steps=10, # 建议设大一点，0.1 step 没意义
    save_strategy="steps",
    save_steps=50,    # 同上
    lr_scheduler_type='linear',
    learning_rate=2e-4,
    bf16=True,
    group_by_length=True,
    dataset_text_field='text',
    report_to='none',
    ddp_find_unused_parameters=False, # 防止多卡训练报错
)

trainer = SFTTrainer(
    model=sft_model,      # 传入 Base Model
    train_dataset=formatted_gsm8k,
    peft_config=peft_config, # 传入 Config，SFTTrainer 会自动应用 LoRA
    processing_class=sft_tokenizer,
    args=training_arguments,
)

trainer.train()

```

---

### 方案二：使用 QLoRA（4-bit 量化，极度节省显存）

如果方案一依然报错 OOM（说明两张卡加起来显存都不够，或者碎片化严重），你需要使用 QLoRA。这将把模型权重压缩到 4-bit（7B 模型仅需约 5GB 显存），从而留出大量空间给梯度和训练。

**需要安装 bitsandbytes:** `pip install bitsandbytes`

**修改后的代码：**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_path = "/home/yxlin/huggingface/Qwen2.5-7B-Instruct"

# 1. 定义量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. 加载模型（带量化和自动设备分配）
sft_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    quantization_config=bnb_config, # 应用 4-bit 量化
    device_map="auto",              # 自动分布到多卡
    trust_remote_code=True
)

sft_tokenizer = AutoTokenizer.from_pretrained(model_path)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# ... (training_arguments 和 trainer 部分与方案一相同) ...

```

---

### 方案三：数据并行 (DDP) - 真正的多卡并行训练

如果你希望**加快训练速度**（而不只是把模型塞进显存），你应该使用 DDP（Distributed Data Parallel）。这种方式下，每张显卡都会加载一个完整的模型副本，数据被拆分并行处理。

**前提**：你的单张显卡显存必须足够大，能放得下整个模型。如果单卡 OOM，则无法使用此方法（除非配合 DeepSpeed）。

**操作步骤：**

1. **不要**在代码中使用 `device_map="auto"`。让 Trainer 自动处理设备分配。
2. 不要直接运行 `python train.py`。
3. 使用 `accelerate` 启动脚本：

首先配置环境（在终端运行一次）：

```bash
accelerate config
# 跟随指引选择：This machine -> multi-GPU -> ...

```

然后启动训练：

```bash
accelerate launch train.py

```

### 总结建议

| 你的情况 | 推荐方案 | 核心代码修改 |
| --- | --- | --- |
| **单卡能放下模型，但想用双卡加速** | **方案三 (DDP)** | 移除 `device_map`，用 `accelerate launch` 启动 |
| **单卡放不下，双卡凑在一起能放下** | **方案一 (Model Parallel)** | `device_map="auto"` |
| **双卡加起来显存也很紧张** | **方案二 (QLoRA)** | `BitsAndBytesConfig` + `device_map="auto"` |

**如果是刚开始微调，我强烈建议你先尝试方案二（QLoRA + device_map="auto"）**，这是最稳健、最不容易 OOM 的方式。