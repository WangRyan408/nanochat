# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Inference - Quick Start

Step 1: Create your .venv and install uv package manager

Step 2: enter your .venv (source .venv/bin/activate)

Step 3: install dependencies (uv sync)

Step 4: Chat with the model
- `$ python -m scripts.chat_cli -p "Hello, how are you?"`

With RAG: `$ python -m scripts.chat_cli -p "Hello, how are you?" --rag`

To output to file: `$ python -m scripts.chat_cli -p "Hello, how are you?" --output_file output.txt`



## Training - Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. 


You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensate by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.



## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   └── runcpu.sh                   # Small example of how to run on CPU/MPS
├── nanochat
│   ├── __init__.py                 # empty
│   ├── adamw.py                    # Distributed AdamW optimizer
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── configurator.py             # A superior alternative to argparse
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── muon.py                     # Distributed Muon optimizer
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── run1000.sh                      # Train the ~$800 nanochat d32
├── rustbpe                         # Custom Rust BPE tokenizer trainer
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # see for why this even exists
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun.sh                     # Train the ~$100 nanochat d20
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
│   └── test_rustbpe.py
└── uv.lock
```

## Contributing

nanochat is nowhere near finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

Current LLM policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
