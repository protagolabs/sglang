import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, read_jsonl

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


def main(args):
    set_default_backend(select_sglang_backend(args))

    lines = list(read_jsonl(args.data_path))
    num_questions = args.num_questions
    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i))
        labels.append(lines[i]["label"])
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def reasoning_gen(s, question: str):
        s += sgl.user(
            question
        )
        s += sgl.assistant(
            sgl.gen(
                "answer",
            )
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.time()
    states = reasoning_gen.run_batch(
        arguments,
        num_threads=args.parallel,
        progress_bar=True,
        temperature=0.6,
        max_new_tokens=32768,
        top_p=0.95
    )
    latency = time.time() - tic
    print(str.strip(states[i]["answer"].split("\n")[-1]))
    preds = []
    for i in range(len(states)):
        preds.append(str.strip(states[i]["answer"].split("\n")[-1]))

    # print(f"{preds=}")
    # print(f"{labels=}")

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

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
    args = add_common_sglang_args_and_parse(parser)
    main(args)
