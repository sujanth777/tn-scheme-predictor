import React, { useState } from 'react';
import { CheckCircle2, ChevronDown, ChevronUp, AlertCircle, Building2, HelpCircle } from 'lucide-react';

export default function ResultsSection({ results }) {
  const [expandedScheme, setExpandedScheme] = useState(null);
  const [showIneligible, setShowIneligible] = useState(false);

  if (!results) return null;

  const { summary, eligible_schemes = [], ineligible_schemes = [] } = results;

  const toggleExpand = (index) => {
    setExpandedScheme(expandedScheme === index ? null : index);
  };

  return (
    <div className="mt-10 space-y-6">
      <div className="flex items-center justify-between pb-3 border-b border-slate-200">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900">
            Eligibility Evaluation Results
          </h2>
          <p className="text-xs sm:text-sm text-slate-500">
            Automated MultiOutput XGBoost recommendation & SHAP decision breakdown
          </p>
        </div>
      </div>

      {/* Summary Statistics Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-slate-500 tracking-wider">Schemes Evaluated</span>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">{summary.total_evaluated}</div>
          <span className="text-2xs text-slate-400">Total Welfare Programs</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-emerald-800 tracking-wider">Eligible Schemes</span>
          <div className="text-2xl font-extrabold text-emerald-700 mt-1">{summary.eligible_count}</div>
          <span className="text-2xs text-emerald-700">Criteria Satisfied</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-slate-500 tracking-wider">Top Match Score</span>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">{summary.highest_confidence}%</div>
          <span className="text-2xs text-slate-400">Highest Individual Match</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-slate-500 tracking-wider">Not Eligible</span>
          <div className="text-2xl font-extrabold text-slate-500 mt-1">{summary.ineligible_count}</div>
          <span className="text-2xs text-slate-400">Criteria Unmet</span>
        </div>
      </div>

      {/* Eligible Schemes Section */}
      <div>
        <h3 className="text-base sm:text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-emerald-700" />
          Eligible Schemes ({eligible_schemes.length})
        </h3>

        {eligible_schemes.length === 0 ? (
          <div className="p-6 bg-slate-50 rounded-xl border border-slate-200 text-center">
            <AlertCircle className="w-8 h-8 text-amber-500 mx-auto mb-2" />
            <p className="text-sm font-semibold text-slate-700">
              No matching schemes found for this citizen profile based on the model's eligibility thresholds.
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Verify the citizen details above or check the Scheme Directory tab for criteria details.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {eligible_schemes.map((scheme) => {
              const isExpanded = expandedScheme === scheme.index;
              return (
                <div
                  key={scheme.index}
                  className="bg-white rounded-xl border border-slate-200 border-l-4 border-l-emerald-600 p-5 shadow-2xs transition hover:shadow-xs"
                >
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-base sm:text-lg font-bold text-slate-900">
                          {scheme.name}
                        </span>
                        <span className="px-2 py-0.5 text-xs font-semibold bg-slate-100 text-slate-600 rounded border border-slate-200">
                          {scheme.category}
                        </span>
                      </div>
                      <div className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                        <Building2 className="w-3.5 h-3.5" />
                        {scheme.department}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-1 text-xs font-bold bg-emerald-50 text-emerald-800 rounded-full border border-emerald-200">
                        Eligible • {scheme.confidence_pct}% Match
                      </span>
                    </div>
                  </div>

                  <p className="text-xs sm:text-sm text-slate-600 mt-3 leading-relaxed">
                    {scheme.description}
                  </p>

                  {/* Confidence Progress Meter */}
                  <div className="mt-3">
                    <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-emerald-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${scheme.confidence_pct}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Explainability Accordion Trigger */}
                  <div className="mt-4 pt-3 border-t border-slate-100">
                    <button
                      type="button"
                      onClick={() => toggleExpand(scheme.index)}
                      className="text-xs font-bold text-emerald-800 hover:text-emerald-900 flex items-center gap-1 transition"
                    >
                      <HelpCircle className="w-4 h-4" />
                      Why did the model recommend this scheme?
                      {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-3 p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-4">
                        <div>
                          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block mb-2">
                            Top Contributing Decision Factors (SHAP Analysis):
                          </span>
                          <div className="space-y-2">
                            {scheme.factors.map((f, fIdx) => (
                              <div
                                key={fIdx}
                                className={`text-xs p-2.5 rounded-md flex items-center justify-between border ${
                                  f.is_positive
                                    ? 'bg-emerald-50/70 border-emerald-200 text-emerald-900'
                                    : 'bg-rose-50/70 border-rose-200 text-rose-900'
                                }`}
                              >
                                <span className="font-medium">
                                  {f.is_positive ? '▲' : '▼'} {f.impact_text}
                                </span>
                                <span className="font-mono font-bold text-2xs px-2 py-0.5 rounded bg-white/70">
                                  {f.score > 0 ? `+${f.score}` : f.score}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Official Guidelines Checklist */}
                        <div>
                          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block mb-2">
                            Official Scheme Criteria Reference:
                          </span>
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                            <div className="p-2 bg-white rounded border border-slate-200">
                              <span className="text-slate-400 block text-2xs uppercase">Target Gender</span>
                              <span className="font-bold text-slate-800">{scheme.criteria.gender}</span>
                            </div>
                            <div className="p-2 bg-white rounded border border-slate-200">
                              <span className="text-slate-400 block text-2xs uppercase">Age Bracket</span>
                              <span className="font-bold text-slate-800">{scheme.criteria.age_range}</span>
                            </div>
                            <div className="p-2 bg-white rounded border border-slate-200">
                              <span className="text-slate-400 block text-2xs uppercase">Income Ceiling</span>
                              <span className="font-bold text-slate-800">{scheme.criteria.income_limit}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Ineligible Schemes Collapsible Section */}
      <div className="pt-2">
        <button
          type="button"
          onClick={() => setShowIneligible(!showIneligible)}
          className="text-xs sm:text-sm font-bold text-slate-600 hover:text-slate-900 flex items-center gap-1 transition"
        >
          Schemes Not Currently Eligible ({ineligible_schemes.length})
          {showIneligible ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showIneligible && (
          <div className="mt-3 space-y-2">
            {ineligible_schemes.map((scheme) => (
              <div
                key={scheme.index}
                className="p-3 bg-white rounded-lg border border-slate-200 flex items-center justify-between text-xs"
              >
                <div>
                  <span className="font-semibold text-slate-800">{scheme.name}</span>
                  <span className="ml-2 text-2xs text-slate-400">({scheme.category})</span>
                </div>
                <span className="font-mono text-slate-500 font-semibold">{scheme.confidence_pct}% Match</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
