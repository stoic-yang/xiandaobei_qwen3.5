import json, time, urllib.request, pathlib
run=pathlib.Path('/public/home/xdzs2026_c166/codex_runs/r3.2-post-r31-prefill-profile-20260707-2210')
prompt=json.loads(open('/public/home/xdzs2026_c166/testdata/8-16K_throughput.jsonl').readline())['prompt']
payload={
    'model':'Qwen3.5-27B',
    'messages':[{'role':'user','content':prompt}],
    'temperature':0.0,
    'max_tokens':1,
    'stream':False,
}
req=urllib.request.Request('http://127.0.0.1:8001/v1/chat/completions', data=json.dumps(payload).encode(), method='POST', headers={'Content-Type':'application/json'})
t0=time.perf_counter_ns()
with urllib.request.urlopen(req, timeout=600) as r:
    body=r.read()
t1=time.perf_counter_ns()
(run/'profile/prefill_request.json').write_text(json.dumps({'prompt_chars':len(prompt),'wall_ms':(t1-t0)/1e6,'response_bytes':len(body),'response_preview':body[:500].decode('utf-8','replace')}, indent=2))
