import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { Moon, AlertCircle, TrendingUp, Users, Activity, Brain } from 'lucide-react';

// Simple file reading - no backend needed!
// Place JSON files in public/data/ folder

const DATA_DIR = '/data';  // Files go in public/data/

const RISK_COLORS = {
  'LOW': '#2ecc71',
  'MODERATE': '#f39c12',
  'HIGH': '#e74c3c',
  'VERY HIGH': '#c0392b'
};

const Dashboard = () => {
  const [statistics, setStatistics] = useState(null);
  const [allSubjects, setAllSubjects] = useState([]);
  const [modelComparison, setModelComparison] = useState([]);
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
      // Load statistics
      const statsRes = await fetch(`${DATA_DIR}/statistics.json`);
      const statsData = await statsRes.json();
      setStatistics(statsData);

      // Load all subjects (combined)
      const subjectsRes = await fetch(`${DATA_DIR}/all_models_combined.json`);
      const subjectsData = await subjectsRes.json();
      setAllSubjects(subjectsData);

      // Load model comparison
      const compRes = await fetch(`${DATA_DIR}/model_comparison_summary.json`);
      const compData = await compRes.json();
      setModelComparison(compData);

    } catch (error) {
      console.error('Error loading data:', error);
      alert('Failed to load data. Make sure JSON files are in public/data/ folder!');
    } finally {
      setLoading(false);
    }
  };

  // Filter subjects by selected model
  const filteredSubjects = selectedModel === 'all' 
    ? allSubjects 
    : allSubjects.filter(s => s.model === selectedModel);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Moon className="w-16 h-16 text-blue-500 animate-pulse mx-auto mb-4" />
          <p className="text-xl text-gray-600">Loading SomnusGuard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Moon className="w-10 h-10 text-blue-600" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">SomnusGuard</h1>
                <p className="text-sm text-gray-600">Sleep Paralysis Risk Analysis Dashboard</p>
              </div>
            </div>
            <div className="text-sm text-gray-600">
              Last Updated: {statistics?.analysis_date}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-8">
            {['overview', 'subjects', 'comparison'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-2 border-b-2 font-medium text-sm ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'overview' && (
          <OverviewTab statistics={statistics} allSubjects={allSubjects} />
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
// OVERVIEW TAB
// ========================================================================

const OverviewTab = ({ statistics, allSubjects }) => {
  if (!statistics) return <div>Loading...</div>;

  const riskDistData = Object.entries(statistics.overall_stats.risk_distribution).map(([risk, count]) => ({
    name: risk,
    value: count,
    color: RISK_COLORS[risk]
  }));

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatCard
          title="Total Subjects"
          value={statistics.total_subjects}
          icon={Users}
          color="blue"
        />
        <StatCard
          title="Mean Accuracy"
          value={`${statistics.overall_stats.mean_accuracy.toFixed(1)}%`}
          icon={TrendingUp}
          color="green"
        />
        <StatCard
          title="High Risk Subjects"
          value={statistics.overall_stats.high_risk_count}
          icon={AlertCircle}
          color="red"
        />
        <StatCard
          title="Mean Risk Score"
          value={statistics.overall_stats.mean_risk_score.toFixed(1)}
          icon={Activity}
          color="purple"
        />
      </div>

      {/* Model Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {statistics.by_model.map((model) => (
          <div key={model.model_key} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">{model.model_name}</h3>
              <Brain className="w-6 h-6" style={{ color: model.color }} />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Accuracy:</span>
                <span className="font-semibold">{model.mean_accuracy.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Subjects:</span>
                <span className="font-semibold">{model.subjects_analyzed}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">High Risk:</span>
                <span className="font-semibold text-red-600">{model.high_risk_count}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Risk Level Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                dataKey="value"
              >
                {riskDistData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Top Risk Subjects */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Top 10 High Risk Subjects</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={allSubjects.sort((a, b) => b.risk_score - a.risk_score).slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="subject_id" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="risk_score" fill="#e74c3c" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// ========================================================================
// SUBJECTS TAB
// ========================================================================

const SubjectsTab = ({ subjects, selectedModel, setSelectedModel, modelComparison, onSelectSubject }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'risk_score', direction: 'desc' });
  const [filterRisk, setFilterRisk] = useState('ALL');

  // Filter and sort
  let filteredSubjects = subjects;
  if (filterRisk !== 'ALL') {
    filteredSubjects = subjects.filter(s => s.risk_level === filterRisk);
  }

  const sortedSubjects = [...filteredSubjects].sort((a, b) => {
    if (sortConfig.direction === 'asc') {
      return a[sortConfig.key] > b[sortConfig.key] ? 1 : -1;
    }
    return a[sortConfig.key] < b[sortConfig.key] ? 1 : -1;
  });

  const handleSort = (key) => {
    setSortConfig({
      key,
      direction: sortConfig.key === key && sortConfig.direction === 'desc' ? 'asc' : 'desc'
    });
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <h3 className="text-lg font-semibold">
            Subject List ({sortedSubjects.length})
          </h3>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">Model:</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2"
              >
                <option value="all">All Models</option>
                {modelComparison.map(m => (
                  <option key={m.model_key} value={m.model_key}>{m.model_name}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">Risk:</label>
              <select
                value={filterRisk}
                onChange={(e) => setFilterRisk(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2"
              >
                <option value="ALL">All</option>
                <option value="LOW">Low</option>
                <option value="MODERATE">Moderate</option>
                <option value="HIGH">High</option>
                <option value="VERY HIGH">Very High</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {[
                { key: 'subject_id', label: 'Subject ID' },
                { key: 'model', label: 'Model' },
                { key: 'accuracy', label: 'Accuracy' },
                { key: 'risk_level', label: 'Risk Level' },
                { key: 'risk_score', label: 'Risk Score' },
                { key: 'rem_to_wake', label: 'REM→Wake' },
              ].map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => handleSort(key)}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100"
                >
                  {label}
                  {sortConfig.key === key && (
                    <span className="ml-1">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>
                  )}
                </th>
              ))}
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedSubjects.map((subject, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm">{subject.subject_id}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <span className="px-2 py-1 text-xs rounded" 
                        style={{ backgroundColor: modelComparison.find(m => m.model_key === subject.model)?.color + '20' }}>
                    {subject.model.toUpperCase()}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">{subject.accuracy.toFixed(1)}%</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className="px-2 py-1 text-xs rounded font-semibold"
                    style={{
                      backgroundColor: RISK_COLORS[subject.risk_level] + '20',
                      color: RISK_COLORS[subject.risk_level]
                    }}
                  >
                    {subject.risk_level}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">{subject.risk_score.toFixed(0)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">{subject.rem_to_wake}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <button
                    onClick={() => onSelectSubject(subject)}
                    className="text-blue-600 hover:text-blue-900 font-medium"
                  >
                    View Details →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
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
    <div className="space-y-6">
      {/* Model Accuracy Comparison */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Model Accuracy Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={accuracyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Bar dataKey="accuracy" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Comparison Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Subjects</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Mean Accuracy</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Std Dev</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">High Risk</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Mean Risk Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {modelComparison.map((model) => (
              <tr key={model.model_key} className="hover:bg-gray-50">
                <td className="px-6 py-4">
                  <span className="font-semibold" style={{ color: model.color }}>
                    {model.model_name}
                  </span>
                </td>
                <td className="px-6 py-4">{model.subjects_analyzed}</td>
                <td className="px-6 py-4">{model.mean_accuracy.toFixed(2)}%</td>
                <td className="px-6 py-4">{model.std_accuracy.toFixed(2)}%</td>
                <td className="px-6 py-4 text-red-600 font-semibold">{model.high_risk_count}</td>
                <td className="px-6 py-4">{model.mean_risk_score.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ========================================================================
// SUBJECT DETAIL MODAL
// ========================================================================

const SubjectModal = ({ subject, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Subject {subject.subject_id}</h2>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Risk Badge */}
          <div className="mb-6">
            <span
              className="inline-block px-4 py-2 rounded-lg font-bold text-lg"
              style={{
                backgroundColor: RISK_COLORS[subject.risk_level] + '20',
                color: RISK_COLORS[subject.risk_level]
              }}
            >
              {subject.risk_level} RISK
            </span>
            <span className="ml-4 text-gray-600">
              Risk Score: {subject.risk_score.toFixed(1)}/100
            </span>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <MetricBox label="Model" value={subject.model.toUpperCase()} />
            <MetricBox label="Accuracy" value={`${subject.accuracy.toFixed(1)}%`} />
            <MetricBox label="REM→Wake" value={subject.rem_to_wake} />
            <MetricBox label="REM Fragmentation" value={subject.rem_fragmentation} />
            <MetricBox label="REM %" value={`${subject.rem_pct.toFixed(1)}%`} />
            <MetricBox label="Sleep Efficiency" value={`${subject.sleep_efficiency.toFixed(1)}%`} />
          </div>

          {/* Recommendations */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">Clinical Recommendations</h3>
            <ul className="space-y-1 text-sm text-blue-800">
              {subject.risk_score >= 50 && (
                <li>• Consult sleep specialist for evaluation</li>
              )}
              {subject.rem_to_wake >= 8 && (
                <li>• Practice sleep hygiene to reduce REM disruptions</li>
              )}
              {subject.sleep_efficiency < 85 && (
                <li>• Improve sleep environment and schedule</li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

// ========================================================================
// UTILITY COMPONENTS
// ========================================================================

const StatCard = ({ title, value, icon: Icon, color }) => {
  const colors = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    red: 'bg-red-100 text-red-600',
    purple: 'bg-purple-100 text-purple-600'
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${colors[color]}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};

const MetricBox = ({ label, value }) => (
  <div className="bg-gray-50 rounded-lg p-4">
    <p className="text-sm text-gray-600 mb-1">{label}</p>
    <p className="text-2xl font-bold text-gray-900">{value}</p>
  </div>
);

export default Dashboard;