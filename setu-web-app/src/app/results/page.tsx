"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown, Clock, Activity } from "lucide-react";
import stats from "@/data/dashboard-stats.json";

export default function Results() {
  const [showStats, setShowStats] = useState(false);

  const effectData = [
    { 
      name: "Already Correct", 
      change: stats.effect_of_correction.queries_already_correct_mrr_change * 100,
      description: "Over-correction penalty on queries that base retriever got right" 
    },
    { 
      name: "Initially Wrong", 
      change: stats.effect_of_correction.queries_that_were_wrong_mrr_change * 100,
      description: "Improvement on hard queries where base retriever failed"
    }
  ];

  const efficiencyData = [
    { name: "Base Pipeline", latency: stats.efficiency.base_latency_ms },
    { name: "SETU v2", latency: stats.efficiency.setu_latency_ms }
  ];

  return (
    <div className="w-full max-w-5xl mx-auto py-12">
      <div className="mb-16 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight font-display text-text-main">Performance Dashboard</h1>
        <p className="text-text-muted max-w-2xl mx-auto text-lg leading-relaxed">
          SETU's value isn't just in fixing queries—it's in knowing when <em>not</em> to fix them, 
          saving both time and accuracy.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
        
        {/* Chart 1: MRR Effect */}
        <div className="premium-card p-6 md:p-8 flex flex-col h-[450px]">
          <h2 className="text-2xl font-bold text-text-main mb-2 font-display">Effect of Correction</h2>
          <p className="text-text-muted mb-8 text-sm">Change in Mean Reciprocal Rank (MRR)</p>
          
          <div className="flex-1 w-full relative">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={effectData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} tickFormatter={(val) => `${val}%`} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
                <Tooltip 
                  cursor={{ fill: '#f1f5f9' }}
                  contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)' }}
                  formatter={(value: any) => [`${value > 0 ? '+' : ''}${Number(value).toFixed(1)}% MRR`, "Impact"]}
                  itemStyle={{ color: '#0f172a', fontWeight: 'bold' }}
                />
                <Bar dataKey="change" radius={[8, 8, 8, 8]}>
                  {effectData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.change > 0 ? 'var(--color-secondary)' : 'var(--color-accent)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-6 flex items-center justify-center gap-6">
            <div className="flex items-center gap-2">
              <TrendingDown className="text-accent w-4 h-4" />
              <span className="text-sm font-medium text-text-muted">Over-correction Penalty (-16%)</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp className="text-secondary w-4 h-4" />
              <span className="text-sm font-medium text-text-muted">Recovery Gain (+48%)</span>
            </div>
          </div>
        </div>

        {/* Chart 2: Efficiency */}
        <div className="premium-card p-6 md:p-8 flex flex-col h-[450px]">
          <h2 className="text-2xl font-bold text-text-main mb-2 font-display">Computational Efficiency</h2>
          <p className="text-text-muted mb-8 text-sm">Average latency per query (ms)</p>
          
          <div className="flex-1 w-full relative">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={efficiencyData} layout="vertical" margin={{ top: 20, right: 30, left: 40, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} unit="ms" axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
                <YAxis dataKey="name" type="category" stroke="#64748b" tick={{ fill: '#0f172a', fontWeight: 600, fontSize: 12 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
                <Tooltip 
                  cursor={{ fill: '#f1f5f9' }}
                  contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)' }}
                  formatter={(value: any) => [`${value}ms`, "Latency"]}
                  itemStyle={{ color: '#0f172a', fontWeight: 'bold' }}
                />
                <Bar dataKey="latency" radius={[0, 8, 8, 0]} barSize={40}>
                  {efficiencyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#94a3b8' : 'var(--color-primary)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="bg-background p-4 rounded-xl border border-border shadow-sm">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-text-muted uppercase tracking-wide font-bold">Steps</span>
                <Activity className="text-primary w-4 h-4" />
              </div>
              <p className="text-2xl font-mono font-bold text-text-main">
                {stats.efficiency.setu_mean_steps} <span className="text-sm text-text-muted font-sans font-normal">avg</span>
              </p>
            </div>
            <div className="bg-background p-4 rounded-xl border border-border shadow-sm">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-text-muted uppercase tracking-wide font-bold">Time Saved</span>
                <Clock className="text-secondary w-4 h-4" />
              </div>
              <p className="text-2xl font-mono font-bold text-secondary">
                -39<span className="text-sm font-sans font-normal">%</span>
              </p>
            </div>
          </div>
        </div>

      </div>

      {/* Technical Details Toggle */}
      <div className="premium-card overflow-hidden">
        <button 
          onClick={() => setShowStats(!showStats)}
          className="w-full flex items-center justify-between p-6 bg-surface hover:bg-surface-elevated transition-colors"
        >
          <span className="text-xl font-bold text-text-main font-display">Statistical Significance (H1-H10)</span>
          {showStats ? <ChevronUp className="text-text-muted" /> : <ChevronDown className="text-text-muted" />}
        </button>
        
        <AnimatePresence>
          {showStats && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-t border-border bg-background/50"
            >
              <div className="p-6 overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-border text-xs text-text-muted uppercase tracking-wider">
                      <th className="py-4 px-4 font-bold">Hypothesis</th>
                      <th className="py-4 px-4 font-bold">Description</th>
                      <th className="py-4 px-4 font-bold">Result</th>
                      <th className="py-4 px-4 font-bold">p-value</th>
                      <th className="py-4 px-4 font-bold">Effect (ρ)</th>
                    </tr>
                  </thead>
                  <tbody className="text-text-main divide-y divide-border/50">
                    {Object.entries(stats.statistical_significance).map(([key, data]: [string, any]) => (
                      <tr key={key} className="hover:bg-surface transition-colors">
                        <td className="py-4 px-4 font-mono font-bold text-primary uppercase">{key}</td>
                        <td className="py-4 px-4 text-sm text-text-muted font-medium">{data.hypothesis}</td>
                        <td className="py-4 px-4">
                          <span className={`px-2.5 py-1 rounded-lg text-xs font-bold border shadow-sm ${data.supported ? 'bg-secondary/10 text-secondary border-secondary/20' : 'bg-accent/10 text-accent border-accent/20'}`}>
                            {data.supported ? "Supported" : "Rejected"}
                          </span>
                        </td>
                        <td className="py-4 px-4 font-mono text-sm">{data.p_value}</td>
                        <td className="py-4 px-4 font-mono text-sm">{data.rho}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

    </div>
  );
}
