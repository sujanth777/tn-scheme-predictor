import React, { useState } from 'react';
import Header from './components/Header';
import EligibilityForm from './components/EligibilityForm';
import ResultsSection from './components/ResultsSection';
import SchemeDirectory from './components/SchemeDirectory';
import ModelMetrics from './components/ModelMetrics';
import Footer from './components/Footer';
import { predictEligibility } from './api';
import { AlertCircle, ExternalLink, ShieldAlert } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('check');
  const [formData, setFormData] = useState({
    Age: 20,
    Gender: 'Female',
    Income: 120000,
    Student: 'Yes',
    GovtSchoolStudent: 'Yes',
    Farmer: 'No',
    Widow: 'No',
    Disability: 'No',
    WorkingWoman: 'No',
    SC: 'No',
    ST: 'No',
    LandOwner: 'No',
    Pregnant: 'No',
    TNResident: 'Yes',
    NoPermanentHouse: 'No',
    BPL: 'No',
    PrimaryEarnerDeceased: 'No',
    GirlChild: 'No',
    EnrolledTraining: 'No',
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const data = await predictEligibility(formData);
      setResults(data);
      // Smoothly scroll down to results
      setTimeout(() => {
        window.scrollTo({ top: 600, behavior: 'smooth' });
      }, 100);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-8">
        {/* Tab 1: Eligibility Check */}
        {activeTab === 'check' && (
          <div className="space-y-6">
            {/* Disclaimer Banner */}
            <div className="p-4 bg-amber-50 rounded-xl border border-amber-200 text-amber-900 text-xs sm:text-sm flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-bold">Citizen Screening Advisory:</span> This machine learning tool provides automated screening based on state welfare rules. Always confirm exact eligibility and submit applications through designated Tamil Nadu e-Sevai or departmental portals.
              </div>
            </div>

            {error && (
              <div className="p-4 bg-rose-50 text-rose-800 rounded-xl border border-rose-200 text-xs sm:text-sm flex items-center gap-3">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <EligibilityForm
              formData={formData}
              setFormData={setFormData}
              onSubmit={handleSubmit}
              loading={loading}
            />

            <ResultsSection results={results} />
          </div>
        )}

        {/* Tab 2: Scheme Directory */}
        {activeTab === 'schemes' && <SchemeDirectory />}

        {/* Tab 3: Model Performance */}
        {activeTab === 'metrics' && <ModelMetrics />}

        {/* Tab 4: About Platform */}
        {activeTab === 'about' && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 sm:p-8 space-y-6 shadow-2xs">
            <div>
              <h2 className="text-xl sm:text-2xl font-bold text-slate-900 mb-2">About the TN Welfare Eligibility Platform</h2>
              <p className="text-sm text-slate-600 leading-relaxed">
                The Tamil Nadu Scheme Eligibility Predictor is an AI-driven public service prototype built for the PBL Project Review (CS5403 Machine Learning). It automates the discovery of suitable government assistance schemes by analyzing citizen demographic and socio-economic profiles against 20 official Tamil Nadu state welfare schemes.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-100">
              <div>
                <h3 className="text-base font-bold text-slate-800 mb-2">System Architecture & ML Pipeline</h3>
                <ul className="text-xs sm:text-sm text-slate-600 space-y-2 list-disc list-inside">
                  <li><strong>Model:</strong> MultiOutputClassifier wrapping 20 XGBoost gradient boosted estimators.</li>
                  <li><strong>Feature Engineering:</strong> 19 citizen attributes transformed into a 36-dimensional one-hot encoded matrix.</li>
                  <li><strong>Explainability:</strong> SHAP (SHapley Additive exPlanations) TreeExplainer calculates exact local decision weights for every recommended scheme.</li>
                  <li><strong>Performance:</strong> Achieved <strong>99.7% Macro F1</strong> and <strong>99.3% Exact Match Accuracy</strong> across test partitions.</li>
                </ul>
              </div>

              <div>
                <h3 className="text-base font-bold text-slate-800 mb-2">Data & Privacy Disclosures</h3>
                <ul className="text-xs sm:text-sm text-slate-600 space-y-2 list-disc list-inside">
                  <li><strong>Synthetic Training Data:</strong> Modeled on official government rules published in budget speeches and district portals to respect citizen data privacy.</li>
                  <li><strong>Statutory Disclaimer:</strong> Not an official sanctioning body. Final approvals require verified identity documents (Aadhaar, Ration Card, Income Certificate).</li>
                </ul>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-100">
              <h3 className="text-base font-bold text-slate-800 mb-3">Official Tamil Nadu Government Portals</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <a
                  href="https://www.tnesevai.tn.gov.in"
                  target="_blank"
                  rel="noreferrer"
                  className="p-3 bg-slate-50 hover:bg-emerald-50/60 rounded-lg border border-slate-200 transition flex items-center justify-between font-semibold text-slate-800"
                >
                  <span>🏛️ TNeGA / e-Sevai Common Service Centers</span>
                  <ExternalLink className="w-4 h-4 text-slate-400" />
                </a>
                <a
                  href="https://kmut.tn.gov.in"
                  target="_blank"
                  rel="noreferrer"
                  className="p-3 bg-slate-50 hover:bg-emerald-50/60 rounded-lg border border-slate-200 transition flex items-center justify-between font-semibold text-slate-800"
                >
                  <span>👩 Kalaignar Magalir Urimai Thittam (KMUT)</span>
                  <ExternalLink className="w-4 h-4 text-slate-400" />
                </a>
                <a
                  href="https://pudhumaipenn.tn.gov.in"
                  target="_blank"
                  rel="noreferrer"
                  className="p-3 bg-slate-50 hover:bg-emerald-50/60 rounded-lg border border-slate-200 transition flex items-center justify-between font-semibold text-slate-800"
                >
                  <span>🎓 Moovalur Ramamirtham Ammaiyar Pudhumai Penn</span>
                  <ExternalLink className="w-4 h-4 text-slate-400" />
                </a>
                <a
                  href="https://www.naanmudhalvan.tn.gov.in"
                  target="_blank"
                  rel="noreferrer"
                  className="p-3 bg-slate-50 hover:bg-emerald-50/60 rounded-lg border border-slate-200 transition flex items-center justify-between font-semibold text-slate-800"
                >
                  <span>🚀 Naan Mudhalvan Youth Skill Platform</span>
                  <ExternalLink className="w-4 h-4 text-slate-400" />
                </a>
              </div>
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
