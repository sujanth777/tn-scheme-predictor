import React from 'react';
import { User, Briefcase, Users, ShieldAlert, Sparkles, ArrowRight, Loader2 } from 'lucide-react';

const PRESETS = {
  farmer: {
    name: 'Small Cultivator / Farmer',
    data: {
      Age: 42, Gender: 'Male', Income: 180000, Student: 'No',
      GovtSchoolStudent: 'No', Farmer: 'Yes', Widow: 'No',
      Disability: 'No', WorkingWoman: 'No', SC: 'No', ST: 'No',
      LandOwner: 'Yes', Pregnant: 'No', TNResident: 'Yes',
      NoPermanentHouse: 'No', BPL: 'No', PrimaryEarnerDeceased: 'No',
      GirlChild: 'No', EnrolledTraining: 'No',
    },
  },
  student: {
    name: 'College Girl Student',
    data: {
      Age: 20, Gender: 'Female', Income: 120000, Student: 'Yes',
      GovtSchoolStudent: 'Yes', Farmer: 'No', Widow: 'No',
      Disability: 'No', WorkingWoman: 'No', SC: 'No', ST: 'No',
      LandOwner: 'No', Pregnant: 'No', TNResident: 'Yes',
      NoPermanentHouse: 'No', BPL: 'No', PrimaryEarnerDeceased: 'No',
      GirlChild: 'No', EnrolledTraining: 'No',
    },
  },
  senior: {
    name: 'Elderly Resident (BPL)',
    data: {
      Age: 68, Gender: 'Male', Income: 45000, Student: 'No',
      GovtSchoolStudent: 'No', Farmer: 'No', Widow: 'No',
      Disability: 'No', WorkingWoman: 'No', SC: 'No', ST: 'No',
      LandOwner: 'No', Pregnant: 'No', TNResident: 'Yes',
      NoPermanentHouse: 'Yes', BPL: 'Yes', PrimaryEarnerDeceased: 'No',
      GirlChild: 'No', EnrolledTraining: 'No',
    },
  },
  workingWoman: {
    name: 'BPL Working Mother',
    data: {
      Age: 38, Gender: 'Female', Income: 85000, Student: 'No',
      GovtSchoolStudent: 'No', Farmer: 'No', Widow: 'No',
      Disability: 'No', WorkingWoman: 'Yes', SC: 'No', ST: 'No',
      LandOwner: 'No', Pregnant: 'No', TNResident: 'Yes',
      NoPermanentHouse: 'No', BPL: 'Yes', PrimaryEarnerDeceased: 'No',
      GirlChild: 'Yes', EnrolledTraining: 'No',
    },
  },
};

export default function EligibilityForm({ formData, setFormData, onSubmit, loading }) {
  const handleChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const loadPreset = (key) => {
    if (PRESETS[key]) {
      setFormData(PRESETS[key].data);
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 sm:p-8">
      {/* Quick Fill Presets */}
      <div className="mb-8 p-4 bg-slate-50 rounded-lg border border-slate-200">
        <div className="flex items-center gap-2 mb-2 text-xs font-bold text-slate-700 uppercase tracking-wider">
          <Sparkles className="w-4 h-4 text-emerald-700" />
          Quick-Fill Demo Profiles (For Instant Testing)
        </div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(PRESETS).map(([key, item]) => (
            <button
              key={key}
              type="button"
              onClick={() => loadPreset(key)}
              className="text-xs font-semibold px-3 py-1.5 bg-white border border-slate-300 hover:border-emerald-600 hover:bg-emerald-50 text-slate-700 rounded-md transition-all shadow-2xs"
            >
              👤 {item.name}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={onSubmit} className="space-y-8">
        {/* Section 1: Personal Information */}
        <div>
          <div className="flex items-center gap-2 pb-2 mb-4 border-b border-slate-100 text-slate-900 font-bold text-base sm:text-lg">
            <User className="w-5 h-5 text-emerald-800" />
            <span>1. Personal Information</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Age (Years)</label>
              <input
                type="number"
                min="1"
                max="100"
                value={formData.Age}
                onChange={(e) => handleChange('Age', parseInt(e.target.value) || 1)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Gender</label>
              <select
                value={formData.Gender}
                onChange={(e) => handleChange('Gender', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Tamil Nadu Resident</label>
              <select
                value={formData.TNResident}
                onChange={(e) => handleChange('TNResident', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Currently a Student</label>
              <select
                value={formData.Student}
                onChange={(e) => handleChange('Student', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Studied in Government School</label>
              <select
                value={formData.GovtSchoolStudent}
                onChange={(e) => handleChange('GovtSchoolStudent', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 2: Socio-Economic & Occupation */}
        <div>
          <div className="flex items-center gap-2 pb-2 mb-4 border-b border-slate-100 text-slate-900 font-bold text-base sm:text-lg">
            <Briefcase className="w-5 h-5 text-emerald-800" />
            <span>2. Socio-Economic & Occupation</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Annual Household Income (INR)</label>
              <input
                type="number"
                min="0"
                max="1000000"
                step="5000"
                value={formData.Income}
                onChange={(e) => handleChange('Income', parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition"
                required
              />
              <span className="text-2xs text-slate-500 mt-1 block">₹{formData.Income.toLocaleString('en-IN')}</span>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Below Poverty Line (BPL)</label>
              <select
                value={formData.BPL}
                onChange={(e) => handleChange('BPL', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Farmer / Cultivator</label>
              <select
                value={formData.Farmer}
                onChange={(e) => handleChange('Farmer', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Agricultural Land Owner</label>
              <select
                value={formData.LandOwner}
                onChange={(e) => handleChange('LandOwner', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Working Woman</label>
              <select
                value={formData.WorkingWoman}
                onChange={(e) => handleChange('WorkingWoman', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Enrolled in Skill Training</label>
              <select
                value={formData.EnrolledTraining}
                onChange={(e) => handleChange('EnrolledTraining', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 3: Social Category */}
        <div>
          <div className="flex items-center gap-2 pb-2 mb-4 border-b border-slate-100 text-slate-900 font-bold text-base sm:text-lg">
            <Users className="w-5 h-5 text-emerald-800" />
            <span>3. Social Category</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Scheduled Caste (SC)</label>
              <select
                value={formData.SC}
                onChange={(e) => handleChange('SC', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Scheduled Tribe (ST)</label>
              <select
                value={formData.ST}
                onChange={(e) => handleChange('ST', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 4: Special Circumstances */}
        <div>
          <div className="flex items-center gap-2 pb-2 mb-4 border-b border-slate-100 text-slate-900 font-bold text-base sm:text-lg">
            <ShieldAlert className="w-5 h-5 text-emerald-800" />
            <span>4. Special Circumstances</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Differently Abled (Disability)</label>
              <select
                value={formData.Disability}
                onChange={(e) => handleChange('Disability', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Widow</label>
              <select
                value={formData.Widow}
                onChange={(e) => handleChange('Widow', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Pregnant / Lactating</label>
              <select
                value={formData.Pregnant}
                onChange={(e) => handleChange('Pregnant', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">No Permanent House (Kutcha)</label>
              <select
                value={formData.NoPermanentHouse}
                onChange={(e) => handleChange('NoPermanentHouse', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Primary Breadwinner Deceased</label>
              <select
                value={formData.PrimaryEarnerDeceased}
                onChange={(e) => handleChange('PrimaryEarnerDeceased', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">Family with Girl Child</label>
              <select
                value={formData.GirlChild}
                onChange={(e) => handleChange('GirlChild', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:outline-hidden transition bg-white"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>
        </div>

        {/* Primary Action Button */}
        <div className="pt-4">
          <button
            type="submit"
            disabled={loading}
            className="w-full py-3.5 px-6 rounded-lg bg-emerald-800 hover:bg-emerald-900 text-white font-bold text-base shadow-sm hover:shadow transition flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Evaluating Citizen Profile Across 20 Schemes...
              </>
            ) : (
              <>
                Check Eligibility
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
