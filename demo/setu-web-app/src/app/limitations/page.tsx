"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Info } from "lucide-react";

export default function Limitations() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="w-full max-w-4xl mx-auto py-12">
      <div className="mb-12 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight font-display text-text-main">System Limitations</h1>
        <p className="text-text-muted max-w-2xl mx-auto text-lg leading-relaxed">
          Transparent reporting of the boundaries and failure modes of the SETU pipeline.
        </p>
      </div>

      <motion.div 
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-6"
      >
        <motion.div variants={item} className="premium-card p-8 relative overflow-hidden group border-l-4 border-l-accent hover:shadow-md transition-shadow">
          <h2 className="text-xl font-bold text-text-main mb-3 flex items-center gap-2 font-display">
            <AlertTriangle className="text-accent" size={20} />
            H1 Not Supported (BGE-M3 Resilience)
          </h2>
          <p className="text-text-muted leading-relaxed">
            The initial hypothesis (H1) assumed that code-mixing drastically degrades search accuracy. However, modern multilingual models like BGE-M3 are highly resilient out-of-the-box. Many Hinglish queries succeed without any intervention, making the "correction" problem much narrower than anticipated.
          </p>
        </motion.div>

        <motion.div variants={item} className="premium-card p-8 relative overflow-hidden group border-l-4 border-l-secondary hover:shadow-md transition-shadow">
          <h2 className="text-xl font-bold text-text-main mb-3 flex items-center gap-2 font-display">
            <Info className="text-secondary" size={20} />
            Constrained Corpus Size
          </h2>
          <p className="text-text-muted leading-relaxed">
            The current evaluation uses a highly constrained corpus of 380 chunks. While excellent for controlled diagnosis of the PM-KISAN and RBI domains, it may not perfectly represent the noise and scale of a massive production database (e.g., millions of documents).
          </p>
        </motion.div>

        <motion.div variants={item} className="premium-card p-8 relative overflow-hidden group border-l-4 border-l-primary hover:shadow-md transition-shadow">
          <h2 className="text-xl font-bold text-text-main mb-3 flex items-center gap-2 font-display">
            <Info className="text-primary" size={20} />
            Language Detection Proxies
          </h2>
          <p className="text-text-muted leading-relaxed">
            The system currently relies on heuristics (ASCII fast-path) for estimating the Code-Mixing Index (CMI) and Language Identification (LID) entropy. While performant and effective for the bandit state representation, it is a simplification compared to running heavy token-level LID models.
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}
