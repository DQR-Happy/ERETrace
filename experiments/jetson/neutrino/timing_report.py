"""Validate and summarize stage JSON: python -m neutrino.timing_report FILE ..."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(documents):
    rows = defaultdict(lambda: dict(calls=0, sampled=0, errors=0, host_ns=0, gpu_ns=0,
                                   trace_bytes=0, d2h_bytes=0, stages=defaultdict(int)))
    for doc in documents:
        if doc['schema_version'] != 1:
            raise ValueError('Unsupported timing schema')
        if sum(g['calls'] for g in doc['groups']) != doc['calls_completed']:
            raise ValueError('Completed launch counts do not reconcile')
        if sum(g['sampled'] for g in doc['groups']) != doc['calls_sampled']:
            raise ValueError('Sampled launch counts do not reconcile')
        for g in doc['groups']:
            host = g['host_wall']['total_ns']
            if sum(s['total_ns'] for s in g['stages'].values()) != host:
                raise ValueError('Stage partition does not reconcile: ' + g['kernel'])
            if g['sampled'] != g['host_wall']['samples']:
                raise ValueError('Sample counts do not reconcile')
            row = rows[(g['outcome'], g['cold'])]
            for target, source in [('calls', 'calls'), ('sampled', 'sampled'), ('errors', 'errors'),
                                   ('trace_bytes', 'sampled_trace_bytes'), ('d2h_bytes', 'sampled_d2h_bytes')]:
                row[target] += g[source]
            row['host_ns'] += host
            row['gpu_ns'] += g['gpu_probe_interval']['total_ns']
            for stage, stat in g['stages'].items():
                row['stages'][stage] += stat['total_ns']
    return rows


def render(documents, paths):
    rows = summarize(documents)
    sampled = any(d['sample_every'] != 1 for d in documents)
    lines = ['# Neutrino 阶段计时', '',
             'Host 阶段互斥；GPU Event 区间与 host 时间重叠，不能相加。', '',
             ('当前启用抽样：以下时间和字节只统计实测样本，未外推为完整工作负载。' if sampled else
              '当前逐次计时：所有已完成、被 hook 观察到的 launch 均参与汇总。'), '',
             '| 路径 | 编译缓存 | 调用数 | 样本数 | 错误数 | host 总计 ms | GPU 探针区间 ms |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for (outcome, cold), r in sorted(rows.items()):
        lines.append(f'| {outcome} | {"cold" if cold else "warm"} | {r["calls"]} | {r["sampled"]} | {r["errors"]} | {r["host_ns"]/1e6:.3f} | {r["gpu_ns"]/1e6:.3f} |')
    for (outcome, cold), r in sorted(rows.items()):
        lines += ['', f'## {outcome} / {"cold" if cold else "warm"}', '',
                  '| 阶段 | 实测总计 ms | 占该组 host 时间 | 每样本摊销 µs |',
                  '| --- | ---: | ---: | ---: |']
        for stage, ns in sorted(r['stages'].items(), key=lambda x: -x[1]):
            share = 100 * ns / r['host_ns'] if r['host_ns'] else 0
            mean = ns / r['sampled'] / 1e3 if r['sampled'] else 0
            lines.append(f'| {stage} | {ns/1e6:.3f} | {share:.2f}% | {mean:.3f} |')
    lines += ['', '## 测量边界', '',
              '- 多 host 线程的 hook 时长可能重叠；总和不等于应用墙钟耗时。',
              '- `original_submit` 是异步提交耗时；不代表原始 kernel 的 GPU 执行时间。',
              '- `trace_payload_write` 与 `trace_close` 包含 libc/文件系统开销，没有执行 fsync。',
              '- `cold_python_process` 包含解释器启动、编译及子进程计时文件写入；子阶段不可再与其相加。',
              '- 正常退出和 explicit 是完整调用边界快照；periodic 可能缺少最后一段或未结束的调用。', '',
              '| 文件 | 快照 | 聚合开销 ms | 此前快照开销 ms | 初始化 ms |',
              '| --- | --- | ---: | ---: | ---: |']
    for p, d in zip(paths, documents):
        lines.append(f'| {p} | {d["snapshot_kind"]} | {d["accounting_ns"]/1e6:.3f} | {d["previous_snapshot_ns"]/1e6:.3f} | {d["init_ns"]/1e6:.3f} |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    text = render([json.loads(p.read_text()) for p in args.files], args.files)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
