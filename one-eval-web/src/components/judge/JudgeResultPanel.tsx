import React, { useState, useEffect } from 'react';

interface JudgeResultPanelProps {
  apiBaseUrl: string;
  taskId: string;
  lang: 'zh' | 'en';
}

export default function JudgeResultPanel({ apiBaseUrl, taskId, lang }: JudgeResultPanelProps) {
  const tt = (zh: string, en: string) => (lang === 'zh' ? zh : en);

  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'ranking' | 'issues' | 'detail'>('ranking');

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/api/judge/result/${taskId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setSummary(data);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [apiBaseUrl, taskId]);

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-6 text-center text-sm text-slate-400">
        {tt('加载评分结果...', 'Loading judge results...')}
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="bg-white rounded-xl border border-red-200 p-4 text-sm text-red-600">
        {tt('加载失败: ', 'Load failed: ')}{error}
      </div>
    );
  }

  const rankings = summary.comparison?.model_rankings || [];
  const globalIssues = summary.comparison?.global_top_critical_issues || [];
  const globalOther = summary.comparison?.global_top_other_issues || [];

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      {/* Tabs */}
      <div className="flex border-b border-slate-100">
        {([
          ['ranking', tt('模型排名', 'Rankings')],
          ['issues', tt('高频问题', 'Top Issues')],
          ['detail', tt('评分明细', 'Detail')],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as any)}
            className={`px-4 py-2.5 text-xs font-semibold transition-colors ${
              activeTab === key
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-slate-400 hover:text-slate-600'
            }`}
          >
            {label}
          </button>
        ))}
        {/* Download buttons */}
        <div className="ml-auto flex items-center gap-2 pr-3">
          <a
            href={`${apiBaseUrl}/api/judge/result/${taskId}/csv`}
            className="text-xs text-blue-500 hover:text-blue-700"
          >
            CSV
          </a>
          <a
            href={`${apiBaseUrl}/api/judge/result/${taskId}/xlsx`}
            className="text-xs text-blue-500 hover:text-blue-700"
          >
            XLSX
          </a>
        </div>
      </div>

      {/* Tab content */}
      <div className="p-4">
        {activeTab === 'ranking' && (
          <div>
            {rankings.length === 0 ? (
              <p className="text-xs text-slate-400">{tt('暂无排名数据', 'No ranking data')}</p>
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-left">
                    <th className="py-2 px-2 text-slate-500 w-12">{tt('#', '#')}</th>
                    <th className="py-2 px-2 text-slate-500">{tt('模型', 'Model')}</th>
                    <th className="py-2 px-2 text-slate-500">{tt('样本数', 'Samples')}</th>
                    <th className="py-2 px-2 text-slate-500">{tt('错误数', 'Errors')}</th>
                    {rankings[0]?.avg_scores && Object.keys(rankings[0].avg_scores).map(k => (
                      <th key={k} className="py-2 px-2 text-right text-slate-500">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rankings.map((r: any) => (
                    <tr key={r.model_name} className="border-b border-slate-50 hover:bg-slate-50">
                      <td className="py-2 px-2">
                        <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                          r.ranking === 1 ? 'bg-amber-100 text-amber-700' :
                          r.ranking === 2 ? 'bg-slate-100 text-slate-600' :
                          r.ranking === 3 ? 'bg-orange-50 text-orange-600' :
                          'text-slate-400'
                        }`}>
                          {r.ranking}
                        </span>
                      </td>
                      <td className="py-2 px-2 font-medium text-slate-700 truncate max-w-[150px]" title={r.model_name}>
                        {r.model_name}
                      </td>
                      <td className="py-2 px-2 text-slate-500">{r.success_count}/{r.total_samples}</td>
                      <td className="py-2 px-2 text-slate-500">{r.error_count}</td>
                      {r.avg_scores && Object.entries(r.avg_scores).map(([k, v]) => (
                        <td key={k} className="py-2 px-2 text-right font-mono font-semibold text-emerald-600">
                          {typeof v === 'number' ? v.toFixed(2) : String(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {activeTab === 'issues' && (
          <div className="space-y-4">
            {globalIssues.length === 0 && globalOther.length === 0 ? (
              <p className="text-xs text-slate-400">{tt('未发现问题', 'No issues found')}</p>
            ) : (
              <>
                {globalIssues.length > 0 && (
                  <div>
                    <h4 className="text-xs font-bold text-red-600 mb-2">
                      {tt('最严重问题 Top', 'Critical Issues Top')} {globalIssues.length}
                    </h4>
                    <div className="space-y-1">
                      {globalIssues.map((item: any, i: number) => (
                        <div key={i} className="flex items-start gap-2 text-xs">
                          <span className="bg-red-50 text-red-600 rounded px-1.5 py-0.5 font-mono shrink-0">
                            {item.count}
                          </span>
                          <span className="text-slate-600">{item.issue}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {globalOther.length > 0 && (
                  <div>
                    <h4 className="text-xs font-bold text-amber-600 mb-2">
                      {tt('其他问题 Top', 'Other Issues Top')} {globalOther.length}
                    </h4>
                    <div className="space-y-1">
                      {globalOther.map((item: any, i: number) => (
                        <div key={i} className="flex items-start gap-2 text-xs">
                          <span className="bg-amber-50 text-amber-600 rounded px-1.5 py-0.5 font-mono shrink-0">
                            {item.count}
                          </span>
                          <span className="text-slate-600">{item.issue}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {activeTab === 'detail' && (
          <JudgeDetailTab apiBaseUrl={apiBaseUrl} taskId={taskId} lang={lang} />
        )}
      </div>
    </div>
  );
}


function JudgeDetailTab({ apiBaseUrl, taskId, lang }: { apiBaseUrl: string; taskId: string; lang: 'zh' | 'en' }) {
  const tt = (zh: string, en: string) => (lang === 'zh' ? zh : en);
  const [records, setRecords] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const pageSize = 20;

  useEffect(() => {
    fetch(`${apiBaseUrl}/api/judge/detail/${taskId}?limit=0`)
      .then(r => r.json())
      .then(d => setTotal(d.total || 0))
      .catch(() => {});
  }, [apiBaseUrl, taskId]);

  useEffect(() => {
    fetch(`${apiBaseUrl}/api/judge/detail/${taskId}?limit=${pageSize}&offset=${page * pageSize}`)
      .then(r => r.json())
      .then(d => setRecords(d.records || []))
      .catch(() => {});
  }, [apiBaseUrl, taskId, page]);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-slate-400">
          {tt(`共 ${total} 条`, `${total} total`)}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-2 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50"
          >
            {tt('上一页', 'Prev')}
          </button>
          <span className="text-xs text-slate-500">
            {page + 1} / {totalPages || 1}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="px-2 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50"
          >
            {tt('下一页', 'Next')}
          </button>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[60vh] overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-white">
            <tr className="border-b border-slate-200 text-left">
              <th className="py-1.5 px-2 text-slate-500 w-10">#</th>
              <th className="py-1.5 px-2 text-slate-500">{tt('模型', 'Model')}</th>
              <th className="py-1.5 px-2 text-slate-500">{tt('问题', 'Question')}</th>
              <th className="py-1.5 px-2 text-slate-500">{tt('think', 'think')}</th>
              <th className="py-1.5 px-2 text-slate-500">{tt('body', 'body')}</th>
              <th className="py-1.5 px-2 text-slate-500">{tt('严重问题', 'Critical')}</th>
              <th className="py-1.5 px-2 text-slate-500">{tt('备注', 'Remark')}</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r: any, i: number) => (
              <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                <td className="py-1.5 px-2 text-slate-400">{r.sample_index ?? i}</td>
                <td className="py-1.5 px-2 font-medium text-slate-600 truncate max-w-[80px]" title={r.model_name}>
                  {r.model_name}
                </td>
                <td className="py-1.5 px-2 text-slate-500 truncate max-w-[150px]" title={r.question}>
                  {r.question}
                </td>
                <td className="py-1.5 px-2 font-mono text-blue-600">{r.think_score ?? '-'}</td>
                <td className="py-1.5 px-2 font-mono text-blue-600">{r.body_score ?? '-'}</td>
                <td className="py-1.5 px-2 text-red-600 truncate max-w-[120px]" title={r.critical_issue}>
                  {r.critical_issue || '-'}
                </td>
                <td className="py-1.5 px-2 text-slate-500 truncate max-w-[120px]" title={r.remark}>
                  {r.remark || '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
