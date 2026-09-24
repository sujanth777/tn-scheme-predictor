import React, { useState, useEffect } from 'react';
import { Search, Building2, Filter, Loader2, AlertCircle } from 'lucide-react';
import { fetchSchemes } from '../api';

export default function SchemeDirectory() {
  const [schemes, setSchemes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  useEffect(() => {
    fetchSchemes()
      .then((data) => {
        setSchemes(data.schemes || []);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const categories = ['All', ...new Set(schemes.map((s) => s.category))].filter(Boolean);

  const filteredSchemes = schemes.filter((s) => {
    const matchesCat = selectedCategory === 'All' || s.category === selectedCategory;
    const matchesSearch =
      !search ||
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.department.toLowerCase().includes(search.toLowerCase()) ||
      s.description.toLowerCase().includes(search.toLowerCase());
    return matchesCat && matchesSearch;
  });

  if (loading) {
    return (
      <div className="p-12 text-center text-slate-500 flex flex-col items-center justify-center gap-2">
        <Loader2 className="w-6 h-6 animate-spin text-emerald-800" />
        <span className="text-sm">Loading 20 Tamil Nadu Welfare Schemes...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-rose-50 text-rose-800 rounded-lg border border-rose-200 flex items-center gap-3">
        <AlertCircle className="w-5 h-5 flex-shrink-0" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900">Tamil Nadu Welfare Schemes Directory</h2>
        <p className="text-xs sm:text-sm text-slate-500">
          Official eligibility guidelines, target demographics, and income criteria for all 20 evaluated programs
        </p>
      </div>

      {/* Search & Filter Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="sm:col-span-2 relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            placeholder="Search by scheme name, department, or keyword..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden bg-white"
          />
        </div>
        <div className="relative">
          <Filter className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden bg-white"
          >
            {categories.map((c) => (
              <option key={c} value={c}>
                {c === 'All' ? 'All Categories' : c}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="text-xs text-slate-500 font-medium">
        Showing <span className="font-bold text-slate-800">{filteredSchemes.length}</span> of 20 Schemes
      </div>

      {/* Grid of Scheme Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredSchemes.map((scheme) => (
          <div
            key={scheme.index}
            className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs flex flex-col justify-between hover:border-slate-300 transition"
          >
            <div>
              <div className="flex items-start justify-between gap-2 mb-1">
                <h3 className="text-base font-bold text-slate-900">{scheme.name}</h3>
                <span className="px-2 py-0.5 text-xs font-semibold bg-slate-100 text-slate-600 rounded border border-slate-200 flex-shrink-0">
                  {scheme.category}
                </span>
              </div>
              <div className="text-xs text-slate-500 flex items-center gap-1 mb-2">
                <Building2 className="w-3.5 h-3.5" />
                {scheme.department}
              </div>
              <p className="text-xs sm:text-sm text-slate-600 line-clamp-2 mb-4 leading-relaxed">
                {scheme.description}
              </p>
            </div>

            <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-xs space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-500">Target Gender:</span>
                <span className="font-bold text-slate-800">{scheme.target_gender}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Age Range:</span>
                <span className="font-bold text-slate-800">{scheme.min_age} - {scheme.max_age} yrs</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Income Limit:</span>
                <span className="font-bold text-slate-800">{typeof scheme.income_limit === 'number' ? `₹${scheme.income_limit.toLocaleString('en-IN')}` : scheme.income_limit}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">TN Domicile:</span>
                <span className="font-bold text-slate-800">{scheme.tn_resident_required}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
