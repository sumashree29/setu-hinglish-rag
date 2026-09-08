"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import demoQueries from "@/data/demo-queries.json";
import { Search, ShieldAlert, Zap, CheckCircle2, Check, ArrowRight } from "lucide-react";

export default function DemoPage() {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [pipelineState, setPipelineState] = useState(0); 

  const handleSelect = (idx: number) => {
    setSelectedIndex(idx);
    setPipelineState(0);
    
    // Simulate pipeline progression
    setTimeout(() => setPipelineState(1), 800); // Confidence Check
    setTimeout(() => setPipelineState(2), 1800); // Operator applied
    setTimeout(() => setPipelineState(3), 2800); // Result
  };

  const selectedQuery = selectedIndex !== null ? demoQueries[selectedIndex] : null;

  return (
    <div className="w-full max-w-6xl mx-auto">
      <div className="mb-12 text-center pt-8">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight font-display text-text-main">Interactive Demo</h1>
        <p className="text-text-muted max-w-2xl mx-auto text-lg leading-relaxed">
          Select a Hinglish query below to see how SETU adapts its behavior based on the query's complexity. 
          Real queries, real ranks, no invented numbers.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Sidebar: Query Selection */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center gap-2 mb-6 px-2">
            <Search className="text-primary w-5 h-5" />
            <h2 className="text-xl font-bold text-text-main font-display">Test Queries</h2>
          </div>
          
          <div className="space-y-3">
            {demoQueries.map((q, idx) => {
              const isEasy = q.ops_applied[0] === "STOP";
              const isActive = selectedIndex === idx;
              return (
                <button
                  key={idx}
                  onClick={() => handleSelect(idx)}
                  className={`w-full text-left p-5 rounded-2xl border transition-all duration-300 relative overflow-hidden group ${
                    isActive 
                      ? "bg-surface border-primary/50 shadow-[0_10px_30px_var(--color-primary-glow)] -translate-y-0.5" 
                      : "premium-card hover:border-text-dim hover:-translate-y-0.5"
                  }`}
                >
                  {isActive && (
                    <motion.div 
                      layoutId="active-bg"
                      className="absolute inset-0 bg-primary/5 z-0"
                    />
                  )}
                  <div className="relative z-10">
                    <div className="flex justify-between items-start gap-4">
                      <p className={`font-medium leading-relaxed ${isActive ? 'text-primary' : 'text-text-muted group-hover:text-text-main'}`}>
                        "{q.query}"
                      </p>
                    </div>
                    <div className="mt-4 flex flex-wrap items-center gap-2 text-xs font-mono">
                      <span className="px-2 py-1 rounded-md bg-background border border-border text-text-muted">
                        CMI: {q.cmi.toFixed(2)}
                      </span>
                      <span className={`px-2 py-1 rounded-md border ${isEasy ? 'bg-secondary/10 border-secondary/20 text-secondary' : 'bg-accent/10 border-accent/20 text-accent'}`}>
                        {isEasy ? "High Confidence" : "Low Confidence"}
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Main Area: Pipeline Visualization */}
        <div className="lg:col-span-7">
          <div className="premium-card p-6 md:p-10 min-h-[700px] h-auto flex flex-col relative bg-surface">
            {!selectedQuery ? (
              <div className="m-auto text-center text-text-muted flex flex-col items-center">
                <div className="w-20 h-20 rounded-2xl bg-surface-elevated border border-border/50 flex items-center justify-center mb-6">
                  <Zap className="w-8 h-8 opacity-40 text-primary" />
                </div>
                <p className="text-lg font-medium text-text-main">Select a query to trace it through the SETU pipeline</p>
                <p className="text-sm mt-2 max-w-sm opacity-60">Watch how the controller evaluates confidence and selectively applies correction operators.</p>
              </div>
            ) : (
              <div className="flex flex-col h-full relative z-10 gap-6">
                
                {/* 0. Original Query */}
                <motion.div 
                  key={`query-${selectedIndex}`}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-2"
                >
                  <h3 className="text-sm font-bold text-text-main mb-2 font-display flex items-center gap-2">
                    <Search className="w-4 h-4 text-text-muted" />
                    Processing Query:
                  </h3>
                  <p className="text-lg text-text-main leading-relaxed p-4 rounded-xl bg-background border border-border/60 shadow-sm font-medium">
                    "{selectedQuery.query}"
                  </p>
                </motion.div>

                {/* 1. Base Retrieval */}
                <motion.div 
                  key={`base-${selectedIndex}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="p-6 rounded-2xl border border-border bg-background shadow-sm"
                >
                  <h3 className="text-xs font-bold text-text-dim uppercase tracking-wider mb-3">1. Base Retrieval</h3>
                  <div className="flex items-center justify-between">
                    <p className="text-lg text-text-main font-semibold">Initial Document Rank</p>
                    <div className={`px-4 py-2 rounded-xl font-mono text-lg font-bold border ${selectedQuery.doc_rank_before > 1 ? 'bg-accent/10 text-accent border-accent/20' : 'bg-secondary/10 text-secondary border-secondary/20'}`}>
                      #{selectedQuery.doc_rank_before}
                    </div>
                  </div>
                </motion.div>

                {/* 2. Controller Decision */}
                <AnimatePresence mode="wait">
                  {pipelineState >= 1 && (
                    <motion.div 
                      key={`controller-${selectedIndex}`}
                      initial={{ opacity: 0, height: 0, y: -20 }}
                      animate={{ opacity: 1, height: "auto", y: 0 }}
                      className="p-6 rounded-2xl border border-primary/20 bg-primary/5 relative overflow-hidden"
                    >
                      <div className="absolute top-0 right-0 p-3 opacity-10">
                        <ShieldAlert className="w-24 h-24 text-primary" />
                      </div>
                      
                      <h3 className="text-xs font-bold text-primary uppercase tracking-wider mb-4 relative z-10">
                        2. SETU Bandit Controller
                      </h3>
                      
                      <div className="relative z-10 min-h-[60px] flex items-center">
                        {pipelineState >= 2 ? (
                          <div className="w-full flex flex-col gap-6">
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                              <div>
                                <p className="text-sm text-text-muted mb-1">Selected Action:</p>
                                <span className="px-4 py-2 rounded-xl bg-surface border border-primary/30 font-mono font-bold text-primary text-lg inline-flex shadow-sm">
                                  {selectedQuery.ops_applied[0]}
                                </span>
                              </div>
                              
                              {selectedQuery.ops_applied[0] === "STOP" ? (
                                <div className="flex items-center gap-2 text-secondary bg-secondary/10 px-4 py-3 rounded-xl border border-secondary/20 w-max shadow-sm">
                                  <CheckCircle2 className="w-5 h-5 shrink-0" /> 
                                  <span className="text-sm font-medium">Confidence high. No correction needed.</span>
                                </div>
                              ) : (
                                <div className="flex items-center gap-2 text-accent bg-accent/10 px-4 py-3 rounded-xl border border-accent/20 w-max shadow-sm">
                                  <Zap className="w-5 h-5 shrink-0" /> 
                                  <span className="text-sm font-medium">Confidence low. Activating {selectedQuery.ops_applied[0]}...</span>
                                </div>
                              )}
                            </div>

                            {selectedQuery.ops_applied[0] !== "STOP" && (
                              <motion.div 
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="p-5 rounded-xl border border-border bg-white shadow-sm flex flex-col gap-3"
                              >
                                <p className="text-xs font-bold text-text-muted uppercase tracking-wider">Generated Query / Embedding</p>
                                <p className="font-mono text-sm text-text-main leading-relaxed bg-surface-elevated p-4 rounded-lg border border-border/50">
                                  {selectedQuery.modified_query}
                                </p>
                              </motion.div>
                            )}
                          </div>
                        ) : (
                          <div className="flex items-center gap-4 text-text-main">
                            <div className="w-5 h-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                            <span className="font-medium">Evaluating confidence and query state...</span>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* 3. Final Output */}
                <AnimatePresence mode="wait">
                  {pipelineState >= 3 && (
                    <motion.div 
                      key={`final-${selectedIndex}`}
                      initial={{ opacity: 0, y: 30, scale: 0.98 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                      className="mt-auto p-6 md:p-8 rounded-2xl bg-surface border border-border shadow-[0_15px_40px_rgb(0,0,0,0.08)] flex-1 flex flex-col relative overflow-hidden"
                    >
                      <div className="absolute top-0 left-0 w-full h-1 bg-secondary" />
                      
                      <div className="flex items-center justify-between mb-6 pb-6 border-b border-border">
                        <h3 className="text-lg font-bold text-text-main flex items-center gap-3">
                          <Check className="text-white w-6 h-6 bg-secondary rounded-full p-1" /> 
                          Final Retrieved Target
                        </h3>
                        
                        <div className="flex items-center gap-3">
                          {selectedQuery.doc_rank_before !== selectedQuery.doc_rank_after && (
                            <>
                              <span className="text-text-muted font-mono line-through opacity-70">#{selectedQuery.doc_rank_before}</span>
                              <ArrowRight className="w-4 h-4 text-text-muted" />
                            </>
                          )}
                          <div className="px-4 py-2 bg-secondary/10 text-secondary border border-secondary/20 rounded-xl text-lg font-mono font-bold">
                            Rank #{selectedQuery.doc_rank_after}
                          </div>
                        </div>
                      </div>
                      
                      <div className="bg-background p-5 rounded-xl border border-border/60 text-sm leading-relaxed text-text-muted w-full">
                        {selectedQuery.doc_title && (
                          <p className="font-semibold text-text-main mb-3 text-base">{selectedQuery.doc_title}</p>
                        )}
                        <p>{selectedQuery.doc_text}</p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
