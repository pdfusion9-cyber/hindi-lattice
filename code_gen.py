#!/usr/bin/env python3
"""code_gen.py -- sawaalon ka ek hissa le, Qwen se jawab le, JSON me save karo.

  python code_gen.py --shard 0 --total 8
"""
import argparse, json, math, os, re, struct, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--total", type=int, default=8)
ap.add_argument("--maxnew", type=int, default=110)
ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.shard)

D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6

# ---------- sawaal bank (200) ----------
Q = [
 "Write a Python function to add two numbers.","Write a Python function to subtract two numbers.",
 "Write a Python function to multiply two numbers.","Write a Python function to divide two numbers.",
 "Write a Python function to find the remainder of division.","Write a Python function to calculate power of a number.",
 "Write a Python function to find the square root of a number.","Write a Python function to find absolute value.",
 "Write a Python function to round a number.","Write a Python function to find the average of two numbers.",
 "Write a Python function to reverse a string.","Write a Python function to check if a string is a palindrome.",
 "Write a Python function to count characters in a string.","Write a Python function to convert string to uppercase.",
 "Write a Python function to convert string to lowercase.","Write a Python function to capitalize first letter.",
 "Write a Python function to count vowels in a string.","Write a Python function to count consonants in a string.",
 "Write a Python function to remove spaces from a string.","Write a Python function to replace a word in a string.",
 "Write a Python function to split a string into words.","Write a Python function to join a list of strings.",
 "Write a Python function to find substring in a string.","Write a Python function to trim whitespace.",
 "Write a Python function to check if a number is even.","Write a Python function to check if a number is odd.",
 "Write a Python function to check if a number is prime.","Write a Python function to find factorial of a number.",
 "Write a Python function to find fibonacci sequence.","Write a Python function to check if number is positive.",
 "Write a Python function to check if number is negative.","Write a Python function to find greatest common divisor.",
 "Write a Python function to find least common multiple.","Write a Python function to sum digits of a number.",
 "Write a Python function to reverse a number.","Write a Python function to count digits in a number.",
 "Write a Python function to check if number is armstrong.","Write a Python function to convert decimal to binary.",
 "Write a Python function to convert binary to decimal.","Write a Python function to find maximum of two numbers.",
 "Write a Python function to find minimum of two numbers.","Write a Python function to find maximum in a list.",
 "Write a Python function to find minimum in a list.","Write a Python function to find sum of a list.",
 "Write a Python function to find average of a list.","Write a Python function to sort a list.",
 "Write a Python function to reverse a list.","Write a Python function to remove duplicates from a list.",
 "Write a Python function to find length of a list.","Write a Python function to check if list is empty.",
 "Write a Python function to merge two lists.","Write a Python function to find common elements in lists.",
 "Write a Python function to find second largest in a list.","Write a Python function to count occurrences in a list.",
 "Write a Python function to filter even numbers from a list.","Write a Python function to filter odd numbers from a list.",
 "Write a Python function to flatten a nested list.","Write a Python function to rotate a list.",
 "Write a Python function to find index of element in list.","Write a Python function to insert element in list.",
 "Write a Python function to remove element from list.","Write a Python function to split list into chunks.",
 "How do I read a file in Python?","How do I write to a file in Python?",
 "How do I append to a file in Python?","How do I read a file line by line in Python?",
 "How do I check if a file exists in Python?","How do I delete a file in Python?",
 "How do I copy a file in Python?","How do I get file size in Python?",
 "How do I create a directory in Python?","How do I list files in a directory in Python?",
 "How do I create a dictionary in Python?","How do I add a key to a dictionary in Python?",
 "How do I delete a key from a dictionary in Python?","How do I iterate over a dictionary in Python?",
 "How do I merge two dictionaries in Python?","How do I sort a dictionary by value in Python?",
 "How do I check if a key exists in a dictionary?","How do I get all keys from a dictionary?",
 "How do I handle exceptions in Python?","How do I raise an exception in Python?",
 "How do I use try except finally in Python?","How do I create a custom exception in Python?",
 "How do I import a module in Python?","How do I install a package in Python?",
 "How do I define a class in Python?","How do I create an object in Python?",
 "How do I use inheritance in Python?","How do I define a method in Python?",
 "How do I use a lambda function in Python?","How do I use map function in Python?",
 "How do I use filter function in Python?","How do I use reduce function in Python?",
 "How do I use list comprehension in Python?","How do I use dictionary comprehension in Python?",
 "How do I check the type of a variable in Python?","How do I convert string to integer in Python?",
 "How do I convert integer to string in Python?","How do I convert list to string in Python?",
 "How do I convert string to list in Python?","How do I check if a string is numeric in Python?",
 "How do I generate a random number in Python?","How do I generate a random string in Python?",
 "How do I get current date in Python?","How do I format a date in Python?",
 "How do I measure execution time in Python?","How do I sleep in Python?",
 "How do I get command line arguments in Python?","How do I read environment variables in Python?",
 "How do I send an HTTP request in Python?","How do I parse JSON in Python?",
 "How do I use regular expressions in Python?","How do I replace text with regex in Python?",
 "How do I remove whitespace from a string?","How do I check if string starts with a prefix?",
 "How do I check if string ends with a suffix?","How do I find the position of a character in a string?",
 "How do I repeat a string in Python?","How do I pad a string with zeros in Python?",
 "How do I compare two strings in Python?","How do I encode a string to bytes in Python?",
 "How do I use enumerate in Python?","How do I use zip in Python?",
 "How do I use sorted with a key in Python?","How do I use any and all in Python?",
 "Write a Python function to swap two variables.","Write a Python function to check leap year.",
 "Write a Python function to convert Celsius to Fahrenheit.","Write a Python function to convert Fahrenheit to Celsius.",
 "Write a Python function to calculate simple interest.","Write a Python function to calculate area of circle.",
 "Write a Python function to calculate area of rectangle.","Write a Python function to calculate perimeter of rectangle.",
 "Write a Python function to find ASCII value of character.","Write a Python function to convert character to ASCII code.",
 "Write a Python function to check if two strings are anagrams.","Write a Python function to find first non repeating character.",
 "Write a Python function to compress a string.","Write a Python function to find longest word in a sentence.",
 "Write a Python function to capitalize each word in a sentence.","Write a Python function to count words in a sentence.",
 "Write a Python function to remove punctuation from a string.","Write a Python function to convert snake case to camel case.",
 "Write a Python function to find duplicate characters in a string.","Write a Python function to check if string contains only digits.",
 "Write a Python function to check if string contains only letters.","Write a Python function to remove vowels from a string.",
 "Write a Python function to find most frequent character.","Write a Python function to sort characters in a string.",
 "Write a Python function to check balanced parentheses.","Write a Python function to implement a stack.",
 "Write a Python function to implement a queue.","Write a Python function to implement binary search.",
 "Write a Python function to implement bubble sort.","Write a Python function to implement insertion sort.",
 "Write a Python function to implement selection sort.","Write a Python function to implement merge sort.",
 "Write a Python function to implement quick sort.","Write a Python function to implement linear search.",
 "Write a Python function to find the median of a list.","Write a Python function to find the mode of a list.",
 "Write a Python function to calculate standard deviation.","Write a Python function to calculate variance.",
 "Write a Python function to find prime numbers in a range.","Write a Python function to find all divisors of a number.",
 "Write a Python function to check if two numbers are coprime.","Write a Python function to convert number to words.",
 "Write a Python function to find sum of squares of numbers.","Write a Python function to find cube of a number.",
 "Write a Python function to check if a list is sorted.","Write a Python function to find missing number in a list.",
 "Write a Python function to find pair with given sum.","Write a Python function to find triplet with given sum.",
 "Write a Python function to move zeros to end of list.","Write a Python function to find intersection of two sets.",
 "Write a Python function to find union of two sets.","Write a Python function to find difference of two sets.",
 "Write a Python function to convert list to tuple.","Write a Python function to convert tuple to list.",
 "Write a Python function to find longest common prefix.","Write a Python function to check if two lists are equal.",
 "Write a Python function to count frequency of elements.","Write a Python function to group elements by key.",
 "Write a Python function to merge sorted lists.","Write a Python function to find kth largest element.",
 "Write a Python function to find kth smallest element.","Write a Python function to remove None from list.",
 "Write a Python function to sum values in a dictionary.","Write a Python function to invert a dictionary.",
 "Write a Python function to find keys with max value.","Write a Python function to sort list of dictionaries.",
 "Write a Python function to validate an email address.","Write a Python function to validate a phone number.",
 "Write a Python function to generate a password.","Write a Python function to hash a string.",
 "Write a Python function to encode base64.","Write a Python function to decode base64.",
 "Write a Python function to calculate compound interest.","Write a Python function to convert rupees to dollars.",
 "Write a Python function to calculate BMI.","Write a Python function to calculate tax.",
 "Write a Python function to countdown timer.","Write a Python function to convert seconds to hours.",
 "Write a Python function to find day of week from date.","Write a Python function to calculate age from birthdate.",
 "Write a Python function to check if date is valid.","Write a Python function to add days to date.",
 "Write a Python function to convert roman to integer.","Write a Python function to convert integer to roman.",
 "Write a Python function to find GCD using recursion.","Write a Python function to find power using recursion.",
 "Write a Python function to print fibonacci using recursion.","Write a Python function to solve tower of hanoi.",
 "Write a Python function to generate all permutations.","Write a Python function to generate all combinations.",
 "Write a Python function to find subsets of a set.","Write a Python function to count paths in a grid.",
]
mine = Q[a.shard::a.total]
print(f"shard {a.shard}/{a.total}: {len(mine)} sawaal", flush=True)

# ---------- wazan ----------
with open("m.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); base = 8 + n

def load(name):
    v = hdr[name]
    with open("m.safetensors", "rb") as f:
        f.seek(base + v["data_offsets"][0])
        raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
    ne = len(raw) // 2
    out = np.empty(ne, dtype=np.float32)
    CH = 2_000_000
    for st in range(0, ne, CH):
        en = min(st + CH, ne)
        u = np.frombuffer(raw[st*2:en*2], dtype=np.uint16)
        out[st:en] = (u.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(out).reshape(v["shape"])

print("wazan...", flush=True)
W = {}
need = ["model.embed_tokens.weight", "model.norm.weight"]
for i in range(L):
    p = f"model.layers.{i}."
    need += [p+"input_layernorm.weight", p+"post_attention_layernorm.weight",
             p+"self_attn.q_proj.weight", p+"self_attn.q_proj.bias",
             p+"self_attn.k_proj.weight", p+"self_attn.k_proj.bias",
             p+"self_attn.v_proj.weight", p+"self_attn.v_proj.bias",
             p+"self_attn.o_proj.weight",
             p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight"]
for nm in need: W[nm] = load(nm)

def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T):
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    half = hd // 2
    x1 = x[..., :half]; x2 = x[..., half:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)

def attn(x, i, T):
    p = f"model.layers.{i}.self_attn."
    q = x @ W[p+"q_proj.weight"].T + W[p+"q_proj.bias"]
    k = x @ W[p+"k_proj.weight"].T + W[p+"k_proj.bias"]
    v = x @ W[p+"v_proj.weight"].T + W[p+"v_proj.bias"]
    q = rope(q.view(T, H, hd), T).transpose(0, 1)
    k = rope(k.view(T, KVH, hd), T).transpose(0, 1)
    v = v.view(T, KVH, hd).transpose(0, 1)
    rep = H // KVH
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    sc = q @ k.transpose(-1, -2) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    sc = sc.masked_fill(mask[None], -1e9)
    o = (sc.softmax(-1) @ v).transpose(0, 1).reshape(T, D)
    return o @ W[p+"o_proj.weight"].T

def mlp(x, i):
    p = f"model.layers.{i}.mlp."
    g = torch.nn.functional.silu(x @ W[p+"gate_proj.weight"].T)
    u = x @ W[p+"up_proj.weight"].T
    return (g * u) @ W[p+"down_proj.weight"].T

def forward(ids):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
SP = {t.content: i for i, t in tok.get_added_tokens_decoder().items()}
IM_S = SP.get("<|im_start|>", 151644); IM_E = SP.get("<|im_end|>", 151645); EOT = SP.get("<|endoftext|>", 151643)

def chat(user):
    ids = [IM_S] + tok.encode("system\nYou are a helpful coding assistant.").ids + [IM_E] + tok.encode("\n").ids
    ids += [IM_S] + tok.encode("user\n"+user).ids + [IM_E] + tok.encode("\n").ids
    ids += [IM_S] + tok.encode("assistant\n").ids
    return ids

def banao(ids, naya, garam=0.5, dand=1.15):
    out = list(ids)
    with torch.no_grad():
        for _ in range(naya):
            lg = forward(out)[-1].clone()
            for t in set(out[-40:]):
                lg[t] = lg[t]/dand if lg[t] > 0 else lg[t]*dand
            nxt = int(torch.multinomial(torch.softmax(lg/garam, -1), 1))
            if nxt in (EOT, IM_E): break
            out.append(nxt)
    return tok.decode(out[len(ids):])

def nikalo_code(txt):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.S)
    if m: return m.group(1).strip()
    lines = [l for l in txt.split("\n") if re.match(r"\s*(def |class |for |if |with |return |import |from )", l)]
    return "\n".join(lines).strip() if len(lines) >= 2 else ""

t0 = time.time()
RES = []
for j, s in enumerate(mine):
    try:
        txt = banao(chat(s), a.maxnew)
        code = nikalo_code(txt)
        if code:
            RES.append({"q": s, "code": code})
        print(f"  [{j+1}/{len(mine)}] {len(code)} ch | {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}", flush=True)

json.dump(RES, open(f"{a.out}/shard{a.shard}.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"SAVED {len(RES)}/{len(mine)} -> {a.out}/shard{a.shard}.json", flush=True)
