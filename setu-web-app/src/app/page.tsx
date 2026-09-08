"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, BookOpen, Search, Code2 } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[85vh] text-center w-full max-w-6xl mx-auto">
      
      {/* Badge */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary font-bold text-sm mb-8 tracking-wide uppercase"
      >
        <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        Research Showcase
      </motion.div>

      {/* Main Heading */}
      <motion.h1 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="text-5xl md:text-7xl font-extrabold tracking-tight text-text-main mb-6 leading-tight max-w-4xl"
      >
        Selective <span className="text-gradient">Correction</span> for Hinglish Retrieval
      </motion.h1>

      {/* Subtitle */}
      <motion.p 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="text-xl md:text-2xl text-text-muted mb-12 max-w-3xl leading-relaxed"
      >
        SETU acts like a triage doctor. It doesn't operate on every query—only the ones that actually need it. Saving compute while boosting accuracy on hard Code-Mixed queries.
      </motion.p>

      {/* CTA Buttons */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto"
      >
        <Link 
          href="/demo" 
          className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-4 rounded-xl bg-primary hover:bg-text-main text-white font-bold text-lg transition-all duration-300 shadow-[0_10px_20px_rgba(154,140,124,0.25)] hover:shadow-[0_15px_30px_rgba(44,42,41,0.2)] hover:-translate-y-0.5"
        >
          <Search size={20} />
          Try Live Demo
        </Link>
        <Link 
          href="/how-it-works" 
          className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-4 rounded-xl bg-surface border border-border/80 hover:border-text-muted text-text-main font-bold text-lg transition-all duration-300 shadow-sm hover:shadow-md hover:-translate-y-0.5"
        >
          <BookOpen size={20} />
          Read the Methodology
        </Link>
      </motion.div>

      {/* Value Props Grid */}
      <motion.div 
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.5 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-24 w-full"
      >
        <div className="premium-card p-8 text-left premium-card-hover group relative overflow-hidden">
          <div className="w-12 h-12 rounded-xl bg-white border border-border shadow-sm flex items-center justify-center mb-6">
            <span className="text-xl font-bold text-primary font-mono">+48%</span>
          </div>
          <h3 className="text-xl font-bold text-text-main mb-3 font-display">Recovery Gain</h3>
          <p className="text-text-muted leading-relaxed">
            Boosts Mean Reciprocal Rank (MRR) by nearly 50% on heavily code-mixed queries where standard retrieval fails.
          </p>
        </div>

        <div className="premium-card p-8 text-left premium-card-hover group relative overflow-hidden">
          <div className="w-12 h-12 rounded-xl bg-white border border-border shadow-sm flex items-center justify-center mb-6">
            <span className="text-xl font-bold text-secondary font-mono">-39%</span>
          </div>
          <h3 className="text-xl font-bold text-text-main mb-3 font-display">Latency Drop</h3>
          <p className="text-text-muted leading-relaxed">
            By avoiding costly LLM calls for easy queries, SETU dramatically reduces overall system latency compared to naive approaches.
          </p>
        </div>

        <div className="premium-card p-8 text-left premium-card-hover group relative overflow-hidden">
          <div className="w-12 h-12 rounded-xl bg-white border border-border shadow-sm flex items-center justify-center mb-6">
            <Code2 className="text-accent" size={24} />
          </div>
          <h3 className="text-xl font-bold text-text-main mb-3 font-display">Bandit Controller</h3>
          <p className="text-text-muted leading-relaxed">
            Uses Contextual Bandits (LinUCB) to dynamically select the cheapest, most effective correction operator per query.
          </p>
        </div>
      </motion.div>

    </div>
  );
}
