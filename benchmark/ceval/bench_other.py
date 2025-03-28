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

choices = ["A", "B", "C", "D"]

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 1] # question在第二列
    k = df.shape[1] - 4
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 2])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 2])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "以下是关于{}的多项选择题（附答案）。\n\n".format(
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
    subjects = sorted(
        [
            f.split("_val.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "val"))
            if "_val.csv" in f
        ]
    )

    all_cors = []
    all_latencies = []
    num_requests = 0

    # Select backend
    call_generate = get_call_generate(args)

    for subject in tqdm(subjects[: args.nsub]):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=0
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "val", subject + "_val.csv"), header=0
        )

        cors, acc, latency = evaluate(args, subject, dev_df, test_df, call_generate)
        all_cors.append(cors)
        all_latencies.append(latency)
        num_requests += len(test_df)

    total_latency = np.sum(all_latencies)
    print("Total latency: {:.3f}".format(total_latency))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "ceval",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(total_latency, 3),
            "accuracy": round(weighted_acc, 3),
            "num_requests": num_requests,
            "other": {
                "nsub": args.nsub,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="sglang/benchmark/ceval/ceval_data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--nsub", type=int, default=60)
    args = add_common_other_args_and_parse(parser)
    main(args)
