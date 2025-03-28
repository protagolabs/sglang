import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

choices = ["true", "false"]

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def get_one_example(lines, i):
    ret = "如下是一个代词消歧的任务。即判断句子中的代词指代的是哪个名词。\n" + \
            "句子：" + lines[i]["text"] + "\n" + \
            "问题：第\"" + str(lines[i]["target"]["span2_index"]) + "\"个字符处开始的代词\"" + lines[i]["target"]["span2_text"] + "\"指代的是第\"" + str(lines[i]["target"]["span1_index"]) + "\"个字符处开始的名词\"" + lines[i]["target"]["span1_text"] + "\"吗？\n" + \
            "请直接给出答案true或者false:"
    return ret


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate(args, subject, dev_df, test_df, call_generate):
    prompts = []
    labels = []

    # Construct prompts
    k = args.ntrain
    train_prompt = gen_prompt(dev_df, subject, k)
    while len(tokenizer.encode(train_prompt)) > 1536:
        k -= 1
        train_prompt = gen_prompt(dev_df, subject, k)

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

        label = test_df.iloc[i, test_df.shape[1] - 1]
        labels.append(label)

    preds = [None] * len(prompts)
    max_tokens = 1

    # Run requests
    if args.backend != "lmql":
        # Use thread pool
        def get_one_answer(i):
            pred = call_generate(prompts[i], temperature=0, max_tokens=max_tokens)
            preds[i] = pred.strip()[0]

        tic = time.time()
        if args.parallel == 1:
            for i in range(len(prompts)):
                get_one_answer(i)
        else:
            with ThreadPoolExecutor(args.parallel) as executor:
                executor.map(get_one_answer, list(range(len(prompts))))
    else:
        # Use asyncio
        async def batched_call(batch_size):
            for i in range(0, len(prompts), batch_size):
                tasks = []
                for p in prompts[i : i + batch_size]:
                    tasks.append(call_generate(p, temperature=0, max_tokens=max_tokens))
                rets = await asyncio.gather(*tasks)
                for j in range(len(rets)):
                    preds[i + j] = rets[j].strip()[0]

        tic = time.time()
        asyncio.run(batched_call(batch_size=args.parallel))
    latency = time.time() - tic

    # Compute accuracy
    cors = [pred == label for pred, label in zip(preds, labels)]
    acc = np.mean(cors)
    cors = np.array(cors)

    print(
        "Average accuracy {:.3f}, latency {:.2f}, #q: {} - {}".format(
            acc, latency, len(prompts), subject
        )
    )

    return cors, acc, latency


def main(args):
    call_generate = get_call_generate(args)

    lines = list(read_jsonl(args.data_path))
    num_questions = args.num_questions
    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i))
        labels.append(lines[i]["label"])
    # arguments = [{"question": q} for q in questions]

    states = [None] * len(labels)

    # Run requests
    if args.backend != "lmql":
        # Use thread pool
        def get_one_answer(i):
            answer = call_generate(
                prompt=questions[i],
                temperature=0,
                max_tokens=256,
                stop=["Question", "Assistant:", "<|separator|>"],
            )
            states[i] = answer

        tic = time.time()
        if args.parallel == 1:
            for i in tqdm(range(len(questions))):
                get_one_answer(i)
        else:
            with ThreadPoolExecutor(args.parallel) as executor:
                list(
                    tqdm(
                        executor.map(get_one_answer, list(range(len(questions)))),
                        total=len(questions),
                    )
                )

    else:
        # Use asyncio
        async def batched_call(batch_size):
            for i in range(0, len(questions), batch_size):
                tasks = []
                for q in questions[i : i + batch_size]:
                    tasks.append(
                        call_generate(
                            q,
                            temperature=0,
                            max_tokens=256,
                            stop="Question",
                        )
                    )
                rets = await asyncio.gather(*tasks)
                for j in range(len(rets)):
                    states[i + j] = rets[j]

        tic = time.time()
        asyncio.run(batched_call(batch_size=args.parallel))
    latency = time.time() - tic

    preds = []
    print(str.strip(states[i]["answer"].split("\n")[-1]))
    for i in range(len(states)):
        preds.append(str.strip(states[i]["answer"].split("\n")[-1]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Latency: {latency:.3f} s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "cluewsc",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", type=str, default="cluewsc_data/train.json")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_other_args_and_parse(parser)
    main(args)
