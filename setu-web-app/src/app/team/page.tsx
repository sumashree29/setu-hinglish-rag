"use client";

import { motion } from "framer-motion";
import { Code2, FileText, User } from "lucide-react";
import Link from "next/link";

export default function Team() {
  return (
    <div className="w-full max-w-4xl mx-auto py-12">
      <div className="mb-12 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight font-display text-text-main">The Team</h1>
        <p className="text-text-muted max-w-2xl mx-auto text-lg leading-relaxed">
          The researchers and developers behind the SETU framework.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="premium-card p-10 text-center flex flex-col items-center hover:shadow-lg transition-shadow duration-300"
        >
          <div className="w-24 h-24 rounded-full bg-surface border border-primary/20 flex items-center justify-center mb-6 shadow-md shadow-primary/10">
            <User size={40} className="text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-text-main mb-1 font-display">Author Name</h2>
          <p className="text-secondary font-medium mb-4 uppercase tracking-widest text-sm">Lead Researcher & Engineer</p>
          <p className="text-text-muted max-w-lg mx-auto leading-relaxed">
            Focusing on Adaptive RAG systems and applied NLP for low-resource and code-mixed languages.
          </p>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4"
        >
          <Link 
            href="#"
            className="premium-card p-8 group hover:border-primary/50 transition-colors flex flex-col items-center text-center premium-card-hover"
          >
            <Code2 size={32} className="text-primary mb-4 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-bold text-text-main mb-2 font-display">View the Repository</h3>
            <p className="text-text-muted text-sm leading-relaxed">Access the full codebase, experimental logs, and data processing scripts.</p>
          </Link>

          <Link 
            href="#"
            className="premium-card p-8 group hover:border-secondary/50 transition-colors flex flex-col items-center text-center premium-card-hover"
          >
            <FileText size={32} className="text-secondary mb-4 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-bold text-text-main mb-2 font-display">Read the Draft</h3>
            <p className="text-text-muted text-sm leading-relaxed">Read the full academic paper detailing the SETU controller and evaluation methodology.</p>
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
