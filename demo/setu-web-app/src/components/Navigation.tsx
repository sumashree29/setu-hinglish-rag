"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrainCircuit, Menu, X, Code2 } from "lucide-react";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function Navigation() {
  const pathname = usePathname();
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { name: "Home", href: "/" },
    { name: "Live Demo", href: "/demo" },
    { name: "How It Works", href: "/how-it-works" },
    { name: "Results", href: "/results" },
  ];

  const dropdownLinks = [
    { name: "Methodology", href: "/methodology" },
    { name: "Limitations", href: "/limitations" },
    { name: "Team", href: "/team" },
  ];

  const DesktopNav = () => (
    <div className="hidden md:flex items-center gap-2 lg:gap-4">
      {navLinks.map((link) => {
        const isActive = pathname === link.href;
        return (
          <Link
            key={link.name}
            href={link.href}
            className={`px-4 py-2 rounded-full text-sm font-semibold transition-all duration-300 relative ${
              isActive 
                ? "text-primary bg-primary/5" 
                : "text-text-muted hover:text-text-main hover:bg-surface-elevated"
            }`}
          >
            {link.name}
            {isActive && (
              <motion.div 
                layoutId="nav-indicator"
                className="absolute inset-0 border border-primary/20 rounded-full z-[-1]"
              />
            )}
          </Link>
        );
      })}
      
      {/* Dropdown for extra pages */}
      <div className="relative group px-2">
        <button className="px-4 py-2 rounded-full text-sm font-semibold text-text-muted hover:text-text-main hover:bg-surface-elevated transition-colors">
          More
        </button>
        <div className="absolute top-full right-0 mt-2 w-48 premium-card opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform origin-top scale-95 group-hover:scale-100 p-2 z-50">
          {dropdownLinks.map((link) => (
            <Link
              key={link.name}
              href={link.href}
              className={`block px-4 py-2.5 text-sm rounded-lg transition-colors ${
                pathname === link.href ? "bg-primary/5 text-primary font-semibold" : "text-text-muted hover:bg-surface-elevated hover:text-text-main"
              }`}
            >
              {link.name}
            </Link>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      isScrolled ? "bg-background/80 backdrop-blur-lg border-b border-border shadow-sm py-3" : "bg-transparent py-5"
    }`}>
      <div className="max-w-6xl mx-auto px-4 lg:px-8 flex items-center justify-between">
        
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center text-white shadow-lg shadow-primary/30 group-hover:scale-105 transition-transform duration-300">
            <BrainCircuit size={22} />
          </div>
          <span className="text-xl font-bold tracking-tight text-text-main font-display">
            SETU <span className="text-primary hidden sm:inline">Framework</span>
          </span>
        </Link>

        {/* Desktop Nav */}
        <DesktopNav />

        {/* Action Button & Mobile Toggle */}
        <div className="flex items-center gap-4">
          <Link 
            href="#" 
            className="hidden sm:flex items-center gap-2 px-5 py-2.5 rounded-full bg-surface border border-border hover:border-primary/50 text-text-main text-sm font-semibold shadow-sm hover:shadow-md transition-all duration-300"
          >
            <Code2 size={18} />
            <span className="hidden lg:inline">View Source</span>
          </Link>
          
          <button 
            className="md:hidden p-2 text-text-main bg-surface-elevated rounded-lg border border-border"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-surface border-b border-border shadow-lg overflow-hidden"
          >
            <div className="px-4 py-6 flex flex-col gap-2">
              {[...navLinks, ...dropdownLinks].map((link) => (
                <Link
                  key={link.name}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`px-4 py-3 rounded-xl text-base font-semibold ${
                    pathname === link.href ? "bg-primary/10 text-primary" : "text-text-muted hover:bg-surface-elevated hover:text-text-main"
                  }`}
                >
                  {link.name}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
