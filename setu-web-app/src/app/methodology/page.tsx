"use client";

import { motion } from "framer-motion";
import { BookOpen, Database, MessageSquare, Percent, BarChart } from "lucide-react";
import stats from "@/data/dashboard-stats.json";

export default function Methodology() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item = {
    hidden: { opacity: 0, x: -20 },
    show: { opacity: 1, x: 0 }
  };

  return (
    <div className="w-full max-w-4xl mx-auto py-12">
      <div className="mb-16 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight font-display text-text-main">Dataset & Methodology</h1>
        <p className="text-text-muted max-w-2xl mx-auto text-lg leading-relaxed">
          The rigorous academic setup behind the SETU evaluation pipeline.
        </p>
      </div>

      <motion.div 
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 gap-6"
      >
        <motion.div variants={item} className="premium-card p-8 group flex items-start gap-6 hover:shadow-md transition-shadow">
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center shrink-0 mt-1 shadow-sm">
            <Database className="text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-text-main mb-2 font-display">Corpus</h2>
            <p className="text-text-muted leading-relaxed">
              Built on a curated set of <strong>{stats.dataset.corpus_chunks} chunks</strong> drawn from official RBI guidelines and PM-KISAN scheme documentation. 
              Provides a highly domain-specific, policy-heavy text base in standard English.
            </p>
          </div>
        </motion.div>

        <motion.div variants={item} className="premium-card p-8 group flex items-start gap-6 hover:shadow-md transition-shadow">
          <div className="w-12 h-12 rounded-xl bg-secondary/10 flex items-center justify-center shrink-0 mt-1 shadow-sm">
            <MessageSquare className="text-secondary" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-text-main mb-2 font-display">Query Generation</h2>
            <p className="text-text-muted leading-relaxed">
              Evaluated on a benchmark of <strong>{stats.dataset.total_queries} queries</strong> systematically generated using GPT-4o. The queries span multiple variants including standard English, pure Hindi (Devanagari), and heavily code-mixed Hinglish.
            </p>
          </div>
        </motion.div>

        <motion.div variants={item} className="premium-card p-8 group flex items-start gap-6 hover:shadow-md transition-shadow">
          <div className="w-12 h-12 rounded-xl bg-accent/10 flex items-center justify-center shrink-0 mt-1 shadow-sm">
            <Percent className="text-accent" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-text-main mb-2 font-display">Code-Mixing Index (CMI)</h2>
            <p className="text-text-muted leading-relaxed">
              The intensity of code-mixing is formally measured using CMI. It calculates the fraction of words belonging to the non-matrix language, capturing the complexity of the Hinglish query.
            </p>
          </div>
        </motion.div>

        <motion.div variants={item} className="premium-card p-8 group flex items-start gap-6 hover:shadow-md transition-shadow">
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center shrink-0 mt-1 border border-primary/20 shadow-sm">
            <BarChart className="text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-text-main mb-2 font-display">Embedding Models & Metrics</h2>
            <p className="text-text-muted leading-relaxed">
              Cross-validated across <strong>{stats.dataset.embedding_models} standard models</strong> (BGE-M3, Indic-SBERT, ME5). System performance is primarily measured using Mean Reciprocal Rank (MRR) and Hit Rate@1.
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
