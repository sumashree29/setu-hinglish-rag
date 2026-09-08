import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-display" });

export const metadata: Metadata = {
  title: "SETU | Adaptive RAG Correction",
  description: "A research showcase for SETU — an adaptive query-correction system for Hinglish search retrieval.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${outfit.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-background text-text-main selection:bg-primary/20 selection:text-primary relative">
        {/* Background glow effects - soft white highlights on the beige background */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
          <div className="absolute -top-[20%] -left-[10%] w-[60%] h-[60%] rounded-full bg-white/60 blur-[120px]" />
          <div className="absolute top-[60%] -right-[10%] w-[50%] h-[60%] rounded-full bg-white/40 blur-[120px]" />
        </div>
        
        <Navigation />
        
        <main className="flex-1 flex flex-col pt-28 pb-16 px-4 relative z-10">
          {children}
        </main>
      </body>
    </html>
  );
}
