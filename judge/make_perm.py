"""Write every cyclic order of each question: shift k puts original option (j+k)%n at position j.
Usage: python make_perm.py test_questions.jsonl test_perm.jsonl"""
import json, sys

def perms(q):
    n = len(q["choices"]); g = ord(q["answer"].strip("()")) - 65
    for k in range(n):
        p = dict(q, id=f'{q["id"]}__s{k}', base_id=q["id"], shift=k,
                 image_paths=[q["image_paths"][(j + k) % n] for j in range(n)],
                 answer=f"({chr(65 + (g - k) % n)})")
        assert p["image_paths"][(g - k) % n] == q["image_paths"][g]  # gold crop is the same in every order
        if "methods" in q: p["methods"] = [q["methods"][(j + k) % n] for j in range(n)]
        yield p

if __name__ == "__main__":
    qs = [json.loads(l) for l in open(sys.argv[1])]
    out = [p for q in qs for p in perms(q)]
    with open(sys.argv[2], "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in out)
    print(len(qs), "questions ->", len(out), "ordered copies")
