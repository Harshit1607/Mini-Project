import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { Moon, AlertCircle, TrendingUp, Users, Activity, Brain, X, Search, Filter } from 'lucide-react';

// ========================================================================
// CONSTANTS & UTILITIES
// ========================================================================

const DATA_DIR = './data';  // Files go in public/data/

const RISK_LEVELS_ORDER = ['LOW', 'MODERATE', 'HIGH', 'VERY HIGH'];

const RISK_COLORS = {
  'LOW': '#10b981', // emerald-600
  'MODERATE': '#f59e0b', // amber-500
  'HIGH': '#ef4444', // red-500
  'VERY HIGH': '#991b1b' // red-800
};

const RISK_GRADIENTS = {
  'LOW': 'from-emerald-500 to-green-600',
  'MODERATE': 'from-amber-500 to-orange-600',
  'HIGH': 'from-red-500 to-rose-600',
  'VERY HIGH': 'from-rose-700 to-red-900'
};

// Utility to wrap risk level in a colored gradient badge
const RiskLevelBadge = ({ level }) => (
    <span
        className={`px-2 py-1 text-xs rounded-lg font-bold bg-gradient-to-r ${RISK_GRADIENTS[level]} text-white shadow-sm`}
    >
        {level}
    </span>
);


// ========================================================================
// UTILITY COMPONENTS (MODERNIZED)
// ========================================================================

const StatCard = ({ title, value, icon: Icon, gradient }) => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 card-hover border border-gray-100 transition-shadow duration-300">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-gray-500 mb-1">{title}</p>
          <p className="text-4xl font-bold text-gray-900 leading-none">{value}</p>
        </div>
        <div className={`p-4 rounded-xl bg-gradient-to-br ${gradient} shadow-xl`}>
          <Icon className="w-7 h-7 text-white" />
        </div>
        </div>
    </div>
  );
};

const MetricBox = ({ label, value, gradient }) => (
  <div className={`bg-gradient-to-br ${gradient} rounded-xl p-4 shadow-lg text-white`}>
    <p className="text-sm font-semibold opacity-90 mb-1">{label}</p>
    <p className="text-2xl font-bold">{value}</p>
  </div>
);

// ========================================================================
// MAIN DASHBOARD COMPONENT
// ========================================================================

const Dashboard = () => {
  const [statistics, setStatistics] = useState(null);
  const [allSubjects, setAllSubjects] = useState([]);
  const [modelComparison, setModelComparison] = useState([]);
  // State to hold the count of unique subject IDs (should be 31)
  const [totalUniqueSubjects, setTotalUniqueSubjects] = useState(0);
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);

  // Load all JSON files on mount
  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    setLoading(true);
    try {
      const statsRes = await fetch(`${DATA_DIR}/statistics.json`);
      const statsData = await statsRes.json();
      setStatistics(statsData);

      const subjectsRes = await fetch(`${DATA_DIR}/all_models_combined.json`);
      const subjectsData = await subjectsRes.json();
      setAllSubjects(subjectsData);

      const compRes = await fetch(`${DATA_DIR}/model_comparison_summary.json`);
      const compData = await compRes.json();
      setModelComparison(compData);

      // FIX: Calculate the true unique subject count
      if (subjectsData && subjectsData.length > 0) {
        const uniqueIds = new Set(subjectsData.map(s => s.subject_id));
        setTotalUniqueSubjects(uniqueIds.size);
      } else if (statsData) {
        setTotalUniqueSubjects(statsData.total_subjects);
      }

    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Filter subjects by selected model (93 entries filtered down to 31 if a model is selected)
  const filteredSubjects = selectedModel === 'all'
    ? allSubjects
    : allSubjects.filter(s => s.model === selectedModel);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center animate-pulse">
          <Moon className="w-16 h-16 text-blue-500 mx-auto mb-4" />
          <p className="text-xl text-gray-600">Loading SomnusGuard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header (Modernized) */}
      <header className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-xl">
                <Moon className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
                  SomnusGuard
                </h1>
                <p className="text-xs text-gray-600">Sleep Paralysis Risk Analysis Dashboard</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-500">Last Updated</p>
              <p className="text-sm text-gray-700 font-semibold">{statistics?.analysis_date}</p>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation (Modernized) */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-6">
            {['overview', 'subjects', 'comparison'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`relative py-3 px-2 font-medium text-sm transition-all duration-300 ${
                  activeTab === tab
                    ? 'text-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                {activeTab === tab && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-600 to-indigo-600"></div>
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'overview' && (
          <OverviewTab
                statistics={statistics}
                allSubjects={allSubjects}
                totalUniqueSubjects={totalUniqueSubjects} // Pass unique count
            />
        )}
        {activeTab === 'subjects' && (
          <SubjectsTab
            subjects={filteredSubjects}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            modelComparison={modelComparison}
            onSelectSubject={setSelectedSubject}
          />
        )}
        {activeTab === 'comparison' && (
          <ComparisonTab modelComparison={modelComparison} allSubjects={allSubjects} />
        )}
      </main>

      {/* Subject Detail Modal */}
      {selectedSubject && (
        <SubjectModal subject={selectedSubject} onClose={() => setSelectedSubject(null)} />
      )}
    </div>
  );
};
// ========================================================================
// OVERVIEW TAB (FIXED DATA AGGREGATION FOR PIE CHART)
// ========================================================================

const OverviewTab = ({ statistics, allSubjects, totalUniqueSubjects }) => {
  if (!statistics) return <div>Loading...</div>;

  const totalSubjects = totalUniqueSubjects;

  // --- START FIX: Aggregate risk only by unique subject ID (Highest Risk Rule) ---
    const riskMap = new Map(); // Key: subject_id, Value: highest RISK_LEVEL_ORDER index
    const riskLevelCounts = { 'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'VERY HIGH': 0 };

    // 1. Determine the highest risk level assigned to each unique subject
    allSubjects.forEach(s => {
        const currentRiskOrder = RISK_LEVELS_ORDER.indexOf(s.risk_level);
        const existingRiskOrder = riskMap.get(s.subject_id) || -1;

        if (currentRiskOrder > existingRiskOrder) {
            riskMap.set(s.subject_id, currentRiskOrder);
        }
    });

    // 2. Count the frequency of the FINAL (Highest) Risk Level
    let totalSubjectsWithRisk = 0;
    riskMap.forEach(order => {
        const finalRiskLevel = RISK_LEVELS_ORDER[order];
        riskLevelCounts[finalRiskLevel]++;
        totalSubjectsWithRisk++;
    });

    // 3. Convert counts to PieChart data format
    const pieChartDataCorrected = RISK_LEVELS_ORDER
        .filter(level => riskLevelCounts[level] > 0) // Only include levels that have counts
        .map(level => {
            const count = riskLevelCounts[level];
            return {
                name: level,
                value: count,
                color: RISK_COLORS[level],
                // Percentage calculated based on the correct total unique subjects (31)
                percentage: ((count / totalSubjects) * 100).toFixed(1)
            };
        });

  return (
    <div className="space-y-12">
      <h2 className="text-3xl font-extrabold text-gray-900 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
          Global Risk Snapshot 🌍
      </h2>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Subjects"
          value={totalSubjects}
          icon={Users}
          gradient="from-blue-500 to-blue-600"
        />
      </div>

      {/* Model Summary Cards */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
          <Brain className="w-6 h-6 mr-3 text-indigo-600" />
          Model Performance Overview
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {statistics.by_model.map((model) => (
            <div key={model.model_key} className="bg-white rounded-xl shadow-xl p-7 card-hover border border-gray-100">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-extrabold text-gray-900">{model.model_name}</h3>
                <Brain className="w-6 h-6" style={{ color: model.color }} />
              </div>
              <div className="space-y-4">
                <div className="p-3 bg-green-50 rounded-lg flex justify-between items-center border-l-4 border-green-400">
                  <span className="text-gray-600 text-sm font-medium">Accuracy</span>
                  <span className="font-extrabold text-3xl text-emerald-600">{model.mean_accuracy.toFixed(1)}%</span>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg flex justify-between items-center border-l-4 border-gray-300">
                  <span className="text-gray-600 text-sm font-medium">Subjects Analyzed</span>
                  <span className="font-bold text-xl text-gray-800">{model.subjects_analyzed}</span>
                </div>
                <div className="p-3 bg-red-50 rounded-lg flex justify-between items-center border-l-4 border-red-400">
                  <span className="text-gray-600 text-sm font-medium">High Risk Count</span>
                  <span className="font-bold text-xl text-red-600">{model.high_risk_count}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Separator */}
      <div className="pt-8 border-t border-gray-200"></div>

      {/* Charts (Fixed Pie Chart Logic) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 pt-4">
        {/* Risk Distribution - NOW CORRECT PERCENTAGES */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Risk Level Distribution (Unique Subjects)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieChartDataCorrected} // <-- USING CORRECTED, UNIQUE-AGGREGATED DATA
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percentage, value }) => `${name}: ${value} (${percentage}%)`}
                outerRadius={110}
                dataKey="value"
                paddingAngle={2}
              >
                {pieChartDataCorrected.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} stroke={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value, name, props) => [`${value} Subjects`, `${props.payload.percentage}% Risk`]} />
              <Legend wrapperStyle={{fontSize: '14px', paddingTop: '10px'}} layout="horizontal" align="center" verticalAlign="bottom" />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Top Risk Subjects */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Top 10 High Risk Subjects</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={allSubjects.sort((a, b) => b.risk_score - a.risk_score).slice(0, 10)} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="subject_id" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11, fill: '#475569' }} />
              <YAxis domain={[0, 100]} label={{ value: 'Risk Score', angle: -90, position: 'insideLeft', offset: 10, fill: '#475569', fontSize: 12 }} />
              <Tooltip formatter={(value) => [`${value.toFixed(1)}/100`, 'Risk Score']} />
              <Bar dataKey="risk_score" fill={RISK_COLORS['HIGH']} radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// ========================================================================
// SUBJECTS TAB (FIXED TO CLUB BY SUBJECT ID)
// ========================================================================

const SubjectsTab = ({ subjects, selectedModel, setSelectedModel, modelComparison, onSelectSubject }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'risk_score', direction: 'desc' });
  const [filterRisk, setFilterRisk] = useState('ALL');

  // --- SHOW ALL 93 ENTRIES (31 subjects × 3 models) ---
  // No consolidation - show all subject-model combinations
  let processedSubjects = subjects;
  // --- END OF PROCESSING ---


  // Filter and sort the PROCESSED list
  let filteredSubjects = processedSubjects;

  if (filterRisk !== 'ALL') {
    filteredSubjects = filteredSubjects.filter(s => s.risk_level === filterRisk);
  }

  const sortedSubjects = [...filteredSubjects].sort((a, b) => {
    const aValue = a[sortConfig.key];
    const bValue = b[sortConfig.key];

    let comparison = 0;
    if (aValue > bValue) comparison = 1;
    else if (aValue < bValue) comparison = -1;

    return sortConfig.direction === 'asc' ? comparison : comparison * -1;
  });

  const handleSort = (key) => {
    setSortConfig({
      key,
      direction: sortConfig.key === key && sortConfig.direction === 'desc' ? 'asc' : 'desc'
    });
  };

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-extrabold text-gray-900 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
          Subject-Level Analysis 🧑‍🤝‍🧑
      </h2>
      {/* Filters (Modernized) */}
      <div className="bg-white rounded-xl shadow p-4 border border-gray-100">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <h3 className="text-xl font-bold text-gray-900">
            Subject List <span className="text-blue-600">({sortedSubjects.length})</span>
          </h3>

          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 w-full sm:w-auto">
            <div className="flex items-center gap-2 w-full sm:w-auto">
              <label className="text-sm font-medium text-gray-600 whitespace-nowrap">Model:</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all w-full sm:w-auto"
              >
                <option value="all">All Models (Unique Subjects)</option>
                {modelComparison.map(m => (
                  <option key={m.model_key} value={m.model_key}>{m.model_name}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2 w-full sm:w-auto">
              <label className="text-sm font-medium text-gray-600 whitespace-nowrap">Risk:</label>
              <select
                value={filterRisk}
                onChange={(e) => setFilterRisk(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all w-full sm:w-auto"
              >
                <option value="ALL">All Risk Levels</option>
                <option value="LOW">Low</option>
                <option value="MODERATE">Moderate</option>
                <option value="HIGH">High</option>
                <option value="VERY HIGH">Very High</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Table (Modernized & Mobile Responsive) */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-100">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-100">
            <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
              <tr>
                {[
                  { key: 'subject_id', label: 'Subject ID' },
                  { key: 'model', label: 'Model' },
                  { key: 'accuracy', label: 'Accuracy (%)' },
                  { key: 'risk_level', label: 'Risk Level' },
                  { key: 'risk_score', label: 'Risk Score' },
                  { key: 'rem_to_wake', label: 'REM→Wake' },
                ].map(({ key, label }) => (
                  <th
                    key={key}
                    onClick={() => handleSort(key)}
                    className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-200 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center space-x-1">
                      <span>{label}</span>
                      {sortConfig.key === key && (
                        <span className="text-blue-600 ml-1">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>
                      )}
                    </div>
                  </th>
                ))}
                <th className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-100">
              {sortedSubjects.map((subject, idx) => (
                <tr key={idx} className="hover:bg-blue-50 transition-colors">
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap text-xs sm:text-sm font-semibold text-gray-900">{subject.subject_id}</td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap text-xs sm:text-sm">
                    <span
                        className="px-2 py-1 text-xs font-bold rounded-lg"
                        style={{
                          backgroundColor: modelComparison.find(m => m.model_key === subject.model)?.color + '20',
                          color: modelComparison.find(m => m.model_key === subject.model)?.color
                        }}
                    >
                        {subject.model.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap text-xs sm:text-sm font-semibold text-emerald-600">{subject.accuracy.toFixed(1)}%</td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap">
                    <RiskLevelBadge level={subject.risk_level} />
                  </td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap text-xs sm:text-sm font-bold text-gray-900">{subject.risk_score.toFixed(0)}</td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap text-xs sm:text-sm text-gray-700">{subject.rem_to_wake}</td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap text-xs sm:text-sm">
                    <button
                      onClick={() => onSelectSubject(subject)}
                      className="text-blue-600 hover:text-blue-800 font-bold transition-all whitespace-nowrap"
                    >
                      <span className="hidden sm:inline">View Details →</span>
                      <span className="sm:hidden">View →</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ========================================================================
// COMPARISON TAB
// ========================================================================

const ComparisonTab = ({ modelComparison, allSubjects }) => {
  const accuracyData = modelComparison.map(m => ({
    name: m.model_name,
    accuracy: m.mean_accuracy,
    fill: m.color
  }));

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-extrabold text-gray-900 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
          Model Performance Comparison 🧠
      </h2>
      {/* Model Accuracy Comparison */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Model Accuracy Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={accuracyData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 12, fontWeight: 600 }} />
            <YAxis domain={[0, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', offset: -10 }} />
            <Tooltip formatter={(value) => [`${value.toFixed(2)}%`, 'Mean Accuracy']} />
            <Bar dataKey="accuracy" radius={[10, 10, 0, 0]}>
                {accuracyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        </div>

      {/* Detailed Comparison Table */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-100">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-100">
            <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
              <tr>
                <th className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider whitespace-nowrap">Model</th>
                <th className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider whitespace-nowrap">Subjects</th>
                <th className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider whitespace-nowrap">Mean Accuracy</th>
                <th className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider whitespace-nowrap">Std Dev</th>
                <th className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider whitespace-nowrap">High Risk</th>
                <th className="px-3 sm:px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider whitespace-nowrap">Mean Risk Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {modelComparison.map((model) => (
                <tr key={model.model_key} className="hover:bg-blue-50 transition-colors">
                  <td className="px-3 sm:px-6 py-3 sm:py-4 font-semibold text-gray-900 whitespace-nowrap text-xs sm:text-sm">
                      <span className="w-2 h-2 rounded-full inline-block mr-2" style={{ backgroundColor: model.color }}></span>
                      {model.model_name}
                  </td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-700 whitespace-nowrap text-xs sm:text-sm">{model.subjects_analyzed}</td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 font-bold text-emerald-600 whitespace-nowrap text-xs sm:text-sm">{model.mean_accuracy.toFixed(2)}%</td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-700 whitespace-nowrap text-xs sm:text-sm">{model.std_accuracy.toFixed(2)}%</td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 whitespace-nowrap">
                      <span className="px-2 py-1 bg-red-100 text-red-700 rounded-lg font-bold text-xs sm:text-sm">
                          {model.high_risk_count}
                      </span>
                  </td>
                  <td className="px-3 sm:px-6 py-3 sm:py-4 font-semibold text-gray-900 whitespace-nowrap text-xs sm:text-sm">{model.mean_risk_score.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      </div>
  );
};

// ========================================================================
// SUBJECT DETAIL MODAL
// ========================================================================
const SubjectModal = ({ subject, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 rounded-t-2xl">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white">Subject {subject.subject_id}</h2>
              <p className="text-blue-100 text-sm mt-1">Detailed Analysis Report</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/20 rounded-full transition-colors"
            >
              <X className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>

        <div className="p-6">
          {/* Risk Badge & Score Progress */}
          <div className="mb-6 flex items-center space-x-4">
            <span
              className={`inline-block px-4 py-2 rounded-xl font-bold text-lg shadow-lg bg-gradient-to-r ${RISK_GRADIENTS[subject.risk_level]} text-white flex-shrink-0`}
            >
              {subject.risk_level} RISK
            </span>
            <div className="flex-1">
              <p className="text-xs text-gray-600 font-medium mb-1">Overall Risk Score</p>
              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-red-500 to-rose-600 transition-all duration-500"
                    style={{ width: `${Math.min(100, subject.risk_score)}%`, borderRadius: '9999px' }}
                  ></div>
                </div>
                <span className="text-sm font-bold text-gray-900">{subject.risk_score.toFixed(1)}/100</span>
              </div>
            </div>
          </div>

          {/* Metrics Grid (Using colorful gradient boxes) */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
            <MetricBox label="Model" value={subject.model.toUpperCase()} gradient="from-purple-500 to-indigo-600" />
            <MetricBox label="Accuracy" value={`${subject.accuracy.toFixed(1)}%`} gradient="from-emerald-500 to-green-600" />
            <MetricBox label="REM→Wake" value={subject.rem_to_wake} gradient="from-blue-500 to-cyan-600" />
            <MetricBox label="REM Fragmentation" value={subject.rem_fragmentation} gradient="from-orange-500 to-amber-600" />
            <MetricBox label="REM %" value={`${subject.rem_pct.toFixed(1)}%`} gradient="from-pink-500 to-rose-600" />
            <MetricBox label="Sleep Efficiency" value={`${subject.sleep_efficiency.toFixed(1)}%`} gradient="from-teal-500 to-emerald-600" />
            </div>

          {/* Recommendations */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-xl p-5">
            <h3 className="font-bold text-blue-900 mb-3 flex items-center">
              <AlertCircle className="w-4 h-4 mr-2" />
              Clinical Recommendations
            </h3>
            <ul className="space-y-2 text-sm">
              {subject.risk_score >= 50 && (
                <li className="text-blue-800">• Consult sleep specialist for evaluation and management plan.</li>
              )}
              {subject.rem_to_wake >= 8 && (
                <li className="text-blue-800">• Practice sleep hygiene to reduce REM disruptions.</li>
              )}
              {subject.sleep_efficiency < 85 && (
                <li className="text-blue-800">• Improve sleep environment and maintain consistent schedule.</li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
