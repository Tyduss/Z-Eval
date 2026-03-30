import React, { useState, useEffect, useRef } from 'react';

interface JudgeProgressPanelProps {
  apiBaseUrl: string;
  taskId: string;
  onComplete: (taskId: string) => void;
  lang: 'zh' | 'en';
}

export default function JudgeProgressPanel({ apiBaseUrl, taskId, onComplete, lang }: JudgeProgressPanelProps) {
  const tt = (zh: string, en: string) => (lang === 'zh' ? zh : en);

  const [status, setStatus] = useState<string>('running');
  const [total, setTotal] = useState(0);
  const [judged, setJudged] = useState(0);
  const [percent, setPercent] = useState(0);
  const [modelsProgress, setModelsProgress] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/api/judge/progress/${taskId}`);
        const data = await res.json();
        setStatus(data.status);
        setTotal(data.total_samples || 0);
        setJudged(data.judged || 0);
        setPercent(data.percent || 0);
        setModelsProgress(data.models_progress || {});

        if (data.status === 'completed') {
          if (intervalRef.current) clearInterval(intervalRef.current);
          onComplete(taskId);
        } else if (data.status === 'failed') {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setError(data.error || 'Unknown error');
        }
      } catch {
        // ignore polling errors
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [apiBaseUrl, taskId, onComplete]);

  const modelEntries = Object.entries(modelsProgress);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-700">
          {status === 'completed'
            ? tt('评分完成', 'Judging Complete')
            : status === 'failed'
              ? tt('评分失败', 'Judging Failed')
              : tt('评分进行中...', 'Judging in progress...')}
        </span>
        <span className="text-xs text-slate-400">{percent}%</span>
      </div>

      {/* Overall progress bar */}
      <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            status === 'completed' ? 'bg-emerald-500' :
            status === 'failed' ? 'bg-red-500' : 'bg-blue-500'
          }`}
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>

      <div className="text-xs text-slate-400">
        {tt(`${judged} / ${total}`, `${judged} / ${total}`)}
      </div>

      {/* Per-model progress */}
      {modelEntries.length > 0 && (
        <div className="space-y-2 border-t border-slate-100 pt-2">
          {modelEntries.map(([name, mp]) => (
            <div key={name} className="flex items-center gap-2">
              <span className="text-xs text-slate-500 truncate max-w-[120px]" title={name}>
                {name}
              </span>
              <div className="flex-1 bg-slate-50 rounded-full h-1.5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    mp.status === 'done' ? 'bg-emerald-400' : 'bg-blue-400'
                  }`}
                  style={{ width: `${mp.percent || 0}%` }}
                />
              </div>
              <span className="text-[10px] text-slate-400 w-12 text-right">
                {mp.status === 'done' ? tt('完成', 'Done') : `${mp.percent || 0}%`}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-600">
          {tt('错误: ', 'Error: ')}{error}
        </div>
      )}
    </div>
  );
}
