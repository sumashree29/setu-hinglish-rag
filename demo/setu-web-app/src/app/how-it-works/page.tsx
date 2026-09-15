"use client";

import { motion } from "framer-motion";
import { ArrowDown, Database, Cpu, ShieldCheck, Zap, Activity } from "lucide-react";

export default function HowItWorks() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="w-full max-w-4xl mx-auto py-12">
      <div className="mb-16 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight font-display text-text-main">How SETU Works</h1>
        <p className="text-text-muted max-w-2xl mx-auto text-lg leading-relaxed">
          Unlike standard Retrieval-Augmented Generation pipelines that blindly rewrite every query, 
          SETU operates like a selective triage system.
        </p>
      </div>

      <motion.div 
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-4"
      >
        {/* Step 1 */}
        <motion.div variants={item} className="premium-card p-8 md:p-10 flex flex-col md:flex-row items-center gap-8 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1.5 h-full bg-primary" />
          <div className="w-16 h-16 shrink-0 rounded-2xl bg-primary/10 flex items-center justify-center text-primary group-hover:scale-110 transition-transform shadow-sm">
            <Database size={32} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-text-main mb-2 font-display">1. Base Retrieval</h2>
            <p className="text-text-muted text-lg">
              The user enters a Hinglish query. SETU immediately runs a fast, standard retrieval against the vector database using the raw query.
            </p>
          </div>
        </motion.div>

        <div className="flex justify-center text-border py-2">
          <ArrowDown className="text-text-dim/50" />
        </div>

        {/* Step 2 */}
        <motion.div variants={item} className="premium-card p-8 md:p-10 flex flex-col md:flex-row items-center gap-8 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1.5 h-full bg-secondary" />
          <div className="w-16 h-16 shrink-0 rounded-2xl bg-secondary/10 flex items-center justify-center text-secondary group-hover:scale-110 transition-transform shadow-sm">
            <Activity size={32} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-text-main mb-2 font-display">2. Confidence Check</h2>
            <p className="text-text-muted text-lg">
              SETU looks at the similarity scores of the top documents. If the margin between the first and subsequent documents is large, it assumes the retrieval succeeded. 
            </p>
            <div className="mt-4 flex gap-4">
              <span className="px-3 py-1.5 rounded-lg bg-secondary/10 text-secondary text-sm font-semibold border border-secondary/20 shadow-sm">High Confidence → Stop</span>
              <span className="px-3 py-1.5 rounded-lg bg-accent/10 text-accent text-sm font-semibold border border-accent/20 shadow-sm">Low Confidence → Intervene</span>
            </div>
          </div>
        </motion.div>

        <div className="flex justify-center text-border py-2">
          <ArrowDown className="text-text-dim/50" />
        </div>

        {/* Step 3 */}
        <motion.div variants={item} className="premium-card p-8 md:p-10 flex flex-col md:flex-row items-center gap-8 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1.5 h-full bg-accent" />
          <div className="w-16 h-16 shrink-0 rounded-2xl bg-accent/10 flex items-center justify-center text-accent group-hover:scale-110 transition-transform shadow-sm">
            <Cpu size={32} />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-text-main mb-2 font-display">3. Selective Operator Application</h2>
            <p className="text-text-muted text-lg mb-4">
              If confidence is low, a Contextual Bandit controller (LinUCB) selects the cheapest, most effective correction operator based on the query's Code-Mixing Index (CMI) and linguistic entropy.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-background border border-border shadow-sm">
                <strong className="text-text-main block mb-1">LAG</strong>
                <span className="text-sm text-text-muted">LLM Augmented Generation (full rewrite)</span>
              </div>
              <div className="p-4 rounded-xl bg-background border border-border shadow-sm">
                <strong className="text-text-main block mb-1">CAEP</strong>
                <span className="text-sm text-text-muted">Entity translation & spelling fix</span>
              </div>
              <div className="p-4 rounded-xl bg-background border border-border shadow-sm">
                <strong className="text-text-main block mb-1">LQP</strong>
                <span className="text-sm text-text-muted">Embedding-space projection</span>
              </div>
            </div>
          </div>
        </motion.div>

        <div className="flex justify-center text-border py-2">
          <ArrowDown className="text-text-dim/50" />
        </div>

        {/* Step 4 */}
        <motion.div variants={item} className="premium-card p-8 md:p-10 flex flex-col md:flex-row items-center gap-8 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1.5 h-full bg-secondary" />
          <div className="w-16 h-16 shrink-0 rounded-2xl bg-secondary/10 flex items-center justify-center text-secondary group-hover:scale-110 transition-transform shadow-sm">
            <ShieldCheck size={32} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-text-main mb-2 font-display">4. Rank Fusion</h2>
            <p className="text-text-muted text-lg">
              Finally, SETU searches the database with the corrected query/embedding. It dynamically fuses the new ranking with the original ranking, weighting the correction higher only if the original query was heavily code-mixed.
            </p>
          </div>
        </motion.div>

      </motion.div>
    </div>
  );
}
