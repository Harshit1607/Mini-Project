"""
SLEEP PARALYSIS RISK ANALYZER
Step 2 of SomnusGuard Pipeline

This module analyzes sleep stage patterns (hypnogram) to detect 
REM instability and other risk factors associated with sleep paralysis.

Authors: Harshit Bareja, Ishika Manchanda, Teena Kaintura
Guide: Ms. Nupur Chugh
Institution: Bharati Vidyapeeth College of Engineering

UPDATED VERSION - Loads predictions from Step 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

class SleepParalysisRiskAnalyzer:
    """
    Analyzes hypnogram (sleep stage sequence) to detect patterns 
    associated with sleep paralysis risk.
    
    Risk Factors Analyzed:
    1. REM Fragmentation - Frequent interruptions of REM sleep
    2. REM-to-Wake Transitions - Direct transitions indicating REM instability
    3. Sleep Architecture Disruption - Overall sleep quality issues
    4. REM Latency - Delayed onset of first REM period
    5. Total REM Percentage - Abnormally high/low REM sleep
    """
    
    def __init__(self, epoch_duration=30):
        """
        Initialize the analyzer
        
        Args:
            epoch_duration: Duration of each epoch in seconds (default: 30)
        """
        self.epoch_duration = epoch_duration
        self.risk_thresholds = {
            'rem_wake_transitions_high': 8,      # >8 REM→Wake transitions is concerning
            'rem_wake_transitions_moderate': 5,  # 5-8 is moderate risk
            'rem_fragmentation_high': 12,        # >12 REM interruptions total
            'rem_fragmentation_moderate': 8,     # 8-12 moderate
            'total_awakenings_high': 20,         # >20 awakenings is poor sleep
            'rem_percentage_low': 15,            # <15% REM is low
            'rem_percentage_high': 30,           # >30% REM is high
            'rem_latency_long': 120,             # >2 hours to first REM (in minutes)
            'stage_transition_rate_high': 25     # >25 transitions per hour
        }
        
    def analyze_hypnogram(self, sleep_stages):
        """
        Main analysis function that processes a hypnogram
        
        Args:
            sleep_stages: Array or list of sleep stage predictions
                         (e.g., [0, 0, 1, 2, 2, 3, 3, 2, 5, 5, 0, ...])
                         Stages: -1=Unknown, 0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM
        
        Returns:
            risk_report: Dictionary containing risk analysis results
        """
        stages = np.array(sleep_stages)
        
        # Calculate all risk factors
        risk_report = {
            'total_epochs': len(stages),
            'duration_hours': len(stages) * self.epoch_duration / 3600,
            'stage_distribution': self._calculate_stage_distribution(stages),
            'rem_analysis': self._analyze_rem_patterns(stages),
            'transition_analysis': self._analyze_transitions(stages),
            'fragmentation_analysis': self._analyze_fragmentation(stages),
            'sleep_architecture': self._analyze_sleep_architecture(stages),
            'risk_factors': [],
            'risk_score': 0,
            'risk_level': 'Low'
        }
        
        # Calculate overall risk score
        risk_report = self._calculate_risk_score(risk_report)
        
        return risk_report
    
    def _calculate_stage_distribution(self, stages):
        """Calculate percentage of time in each sleep stage"""
        stage_counts = Counter(stages)
        total = len(stages)
        
        distribution = {
            'wake_pct': (stage_counts.get(0, 0) / total) * 100,
            'n1_pct': (stage_counts.get(1, 0) / total) * 100,
            'n2_pct': (stage_counts.get(2, 0) / total) * 100,
            'n3_pct': (stage_counts.get(3, 0) / total) * 100,
            'n4_pct': (stage_counts.get(4, 0) / total) * 100,
            'rem_pct': (stage_counts.get(5, 0) / total) * 100,
            'unknown_pct': (stage_counts.get(-1, 0) / total) * 100
        }
        
        # Combine deep sleep stages
        distribution['deep_sleep_pct'] = distribution['n3_pct'] + distribution['n4_pct']
        
        return distribution
    
    def _analyze_rem_patterns(self, stages):
        """Analyze REM sleep patterns and stability"""
        rem_analysis = {
            'rem_periods': [],
            'rem_period_count': 0,
            'rem_latency_minutes': None,
            'longest_rem_period': 0,
            'shortest_rem_period': float('inf'),
            'avg_rem_period_length': 0,
            'rem_fragmentation_index': 0
        }
        
        # Find all REM periods
        in_rem = False
        rem_start = None
        
        for i, stage in enumerate(stages):
            if stage == 5 and not in_rem:  # REM start
                in_rem = True
                rem_start = i
            elif stage != 5 and in_rem:  # REM end
                in_rem = False
                rem_duration = i - rem_start
                rem_analysis['rem_periods'].append({
                    'start_epoch': rem_start,
                    'end_epoch': i,
                    'duration_minutes': (rem_duration * self.epoch_duration) / 60
                })
        
        # Handle case where REM continues to end
        if in_rem:
            rem_duration = len(stages) - rem_start
            rem_analysis['rem_periods'].append({
                'start_epoch': rem_start,
                'end_epoch': len(stages),
                'duration_minutes': (rem_duration * self.epoch_duration) / 60
            })
        
        # Calculate REM metrics
        if rem_analysis['rem_periods']:
            rem_analysis['rem_period_count'] = len(rem_analysis['rem_periods'])
            
            # REM latency (time to first REM)
            first_rem = rem_analysis['rem_periods'][0]['start_epoch']
            rem_analysis['rem_latency_minutes'] = (first_rem * self.epoch_duration) / 60
            
            # Period lengths
            durations = [p['duration_minutes'] for p in rem_analysis['rem_periods']]
            rem_analysis['longest_rem_period'] = max(durations)
            rem_analysis['shortest_rem_period'] = min(durations)
            rem_analysis['avg_rem_period_length'] = np.mean(durations)
            
            # Fragmentation: more periods = more fragmented
            rem_analysis['rem_fragmentation_index'] = len(rem_analysis['rem_periods'])
        
        return rem_analysis
    
    def _analyze_transitions(self, stages):
        """Analyze sleep stage transitions"""
        transition_analysis = {
            'total_transitions': 0,
            'rem_to_wake_count': 0,
            'wake_to_rem_count': 0,
            'rem_to_any_count': 0,
            'transition_rate_per_hour': 0,
            'transition_matrix': {},
            'critical_transitions': []
        }
        
        stage_names = {-1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N4', 5: 'REM'}
        
        # Count transitions
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            if current_stage != next_stage:
                transition_analysis['total_transitions'] += 1
                
                # Track specific transitions
                if current_stage == 5 and next_stage == 0:  # REM → Wake (HIGH RISK)
                    transition_analysis['rem_to_wake_count'] += 1
                    transition_analysis['critical_transitions'].append({
                        'epoch': i,
                        'time_hours': (i * self.epoch_duration) / 3600,
                        'transition': 'REM to Wake'
                    })
                
                if current_stage == 0 and next_stage == 5:  # Wake → REM
                    transition_analysis['wake_to_rem_count'] += 1
                
                if current_stage == 5:  # Any REM exit
                    transition_analysis['rem_to_any_count'] += 1
                
                # Build transition matrix
                trans_key = f"{stage_names.get(current_stage, 'Unknown')} to {stage_names.get(next_stage, 'Unknown')}"
                transition_analysis['transition_matrix'][trans_key] = \
                    transition_analysis['transition_matrix'].get(trans_key, 0) + 1
        
        # Calculate transition rate
        duration_hours = len(stages) * self.epoch_duration / 3600
        transition_analysis['transition_rate_per_hour'] = \
            transition_analysis['total_transitions'] / duration_hours if duration_hours > 0 else 0
        
        return transition_analysis
    
    def _analyze_fragmentation(self, stages):
        """Analyze overall sleep fragmentation"""
        fragmentation = {
            'awakening_count': 0,
            'awakening_index': 0,  # Awakenings per hour
            'sleep_efficiency': 0,
            'wake_after_sleep_onset_minutes': 0
        }
        
        # Find sleep onset (first non-wake epoch after initial wake)
        sleep_onset_idx = None
        for i, stage in enumerate(stages):
            if stage != 0:
                sleep_onset_idx = i
                break
        
        if sleep_onset_idx is not None:
            # Count awakenings after sleep onset
            in_wake = False
            for i in range(sleep_onset_idx, len(stages)):
                if stages[i] == 0 and not in_wake:
                    fragmentation['awakening_count'] += 1
                    in_wake = True
                elif stages[i] != 0:
                    in_wake = False
            
            # Wake after sleep onset
            wake_epochs = sum(1 for s in stages[sleep_onset_idx:] if s == 0)
            fragmentation['wake_after_sleep_onset_minutes'] = \
                (wake_epochs * self.epoch_duration) / 60
            
            # Sleep efficiency (time asleep / time in bed)
            time_in_bed = len(stages) * self.epoch_duration / 60  # minutes
            time_asleep = time_in_bed - fragmentation['wake_after_sleep_onset_minutes']
            fragmentation['sleep_efficiency'] = (time_asleep / time_in_bed * 100) if time_in_bed > 0 else 0
            
            # Awakening index
            duration_hours = len(stages) * self.epoch_duration / 3600
            fragmentation['awakening_index'] = \
                fragmentation['awakening_count'] / duration_hours if duration_hours > 0 else 0
        
        return fragmentation
    
    def _analyze_sleep_architecture(self, stages):
        """Analyze overall sleep architecture quality"""
        architecture = {
            'sleep_cycles_detected': 0,
            'architecture_score': 0,
            'notes': []
        }
        
        # Simplified cycle detection: look for NREM → REM → NREM patterns
        cycles = []
        in_nrem = False
        in_rem = False
        cycle_start = None
        
        for i, stage in enumerate(stages):
            if stage in [1, 2, 3, 4] and not in_nrem:  # NREM start
                in_nrem = True
                if cycle_start is None:
                    cycle_start = i
            elif stage == 5:  # REM
                in_rem = True
            elif stage in [1, 2, 3, 4] and in_rem:  # Back to NREM after REM = cycle complete
                cycles.append(i - cycle_start)
                in_nrem = True
                in_rem = False
                cycle_start = i
        
        architecture['sleep_cycles_detected'] = len(cycles)
        
        # Normal architecture expectations
        expected_cycles = 4-6  # Typical for 7-9 hours sleep
        if len(cycles) < 3:
            architecture['notes'].append("Fewer sleep cycles than expected")
        elif len(cycles) > 7:
            architecture['notes'].append("More sleep cycles than typical")
        
        return architecture
    
    def _calculate_risk_score(self, risk_report):
        """Calculate overall sleep paralysis risk score"""
        risk_factors = []
        risk_score = 0
        
        # Factor 1: REM-to-Wake Transitions (MOST CRITICAL)
        rem_wake_trans = risk_report['transition_analysis']['rem_to_wake_count']
        if rem_wake_trans >= self.risk_thresholds['rem_wake_transitions_high']:
            risk_factors.append({
                'factor': 'High REM-to-Wake Transitions',
                'severity': 'HIGH',
                'value': rem_wake_trans,
                'description': f'{rem_wake_trans} direct REM to Wake transitions detected (threshold: {self.risk_thresholds["rem_wake_transitions_high"]}). This is strongly associated with sleep paralysis risk.',
                'points': 35
            })
            risk_score += 35
        elif rem_wake_trans >= self.risk_thresholds['rem_wake_transitions_moderate']:
            risk_factors.append({
                'factor': 'Moderate REM-to-Wake Transitions',
                'severity': 'MODERATE',
                'value': rem_wake_trans,
                'description': f'{rem_wake_trans} REM to Wake transitions detected (threshold: {self.risk_thresholds["rem_wake_transitions_moderate"]}). Indicates some REM instability.',
                'points': 20
            })
            risk_score += 20
        
        # Factor 2: REM Fragmentation
        rem_frag = risk_report['rem_analysis']['rem_fragmentation_index']
        if rem_frag >= self.risk_thresholds['rem_fragmentation_high']:
            risk_factors.append({
                'factor': 'High REM Fragmentation',
                'severity': 'HIGH',
                'value': rem_frag,
                'description': f'REM sleep occurred in {rem_frag} separate periods (threshold: {self.risk_thresholds["rem_fragmentation_high"]}). Highly fragmented REM is a risk factor.',
                'points': 25
            })
            risk_score += 25
        elif rem_frag >= self.risk_thresholds['rem_fragmentation_moderate']:
            risk_factors.append({
                'factor': 'Moderate REM Fragmentation',
                'severity': 'MODERATE',
                'value': rem_frag,
                'description': f'REM sleep fragmented into {rem_frag} periods (threshold: {self.risk_thresholds["rem_fragmentation_moderate"]}).',
                'points': 15
            })
            risk_score += 15
        
        # Factor 3: Overall Sleep Fragmentation
        awakenings = risk_report['fragmentation_analysis']['awakening_count']
        if awakenings >= self.risk_thresholds['total_awakenings_high']:
            risk_factors.append({
                'factor': 'High Sleep Fragmentation',
                'severity': 'MODERATE',
                'value': awakenings,
                'description': f'{awakenings} awakenings detected (threshold: {self.risk_thresholds["total_awakenings_high"]}). Poor sleep quality contributes to parasomnia risk.',
                'points': 15
            })
            risk_score += 15
        
        # Factor 4: Abnormal REM Percentage
        rem_pct = risk_report['stage_distribution']['rem_pct']
        if rem_pct < self.risk_thresholds['rem_percentage_low']:
            risk_factors.append({
                'factor': 'Low REM Percentage',
                'severity': 'MODERATE',
                'value': f"{rem_pct:.1f}%",
                'description': f'Only {rem_pct:.1f}% REM sleep (normal: 20-25%). REM deprivation can lead to REM rebound and instability.',
                'points': 10
            })
            risk_score += 10
        elif rem_pct > self.risk_thresholds['rem_percentage_high']:
            risk_factors.append({
                'factor': 'High REM Percentage',
                'severity': 'MODERATE',
                'value': f"{rem_pct:.1f}%",
                'description': f'{rem_pct:.1f}% REM sleep (normal: 20-25%). Excessive REM may indicate REM pressure or rebound.',
                'points': 10
            })
            risk_score += 10
        
        # Factor 5: REM Latency
        if risk_report['rem_analysis']['rem_latency_minutes']:
            rem_latency = risk_report['rem_analysis']['rem_latency_minutes']
            if rem_latency > self.risk_thresholds['rem_latency_long']:
                risk_factors.append({
                    'factor': 'Prolonged REM Latency',
                    'severity': 'LOW',
                    'value': f"{rem_latency:.0f} min",
                    'description': f'First REM occurred after {rem_latency:.0f} minutes (typical: 70-90 min). May indicate sleep disruption.',
                    'points': 5
                })
                risk_score += 5
        
        # Factor 6: High Transition Rate
        trans_rate = risk_report['transition_analysis']['transition_rate_per_hour']
        if trans_rate >= self.risk_thresholds['stage_transition_rate_high']:
            risk_factors.append({
                'factor': 'High Stage Transition Rate',
                'severity': 'MODERATE',
                'value': f"{trans_rate:.1f}/hr",
                'description': f'{trans_rate:.1f} stage changes per hour (threshold: {self.risk_thresholds["stage_transition_rate_high"]}/hr). Indicates unstable sleep architecture.',
                'points': 10
            })
            risk_score += 10
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        risk_report['risk_factors'] = risk_factors
        risk_report['risk_score'] = risk_score
        risk_report['risk_level'] = risk_level
        
        return risk_report
    
    def generate_report(self, risk_report, output_dir='./outputs'):
        """
        Generate comprehensive text report
        
        Args:
            risk_report: Risk analysis results
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SLEEP PARALYSIS RISK ANALYSIS REPORT")
        report_lines.append("SomnusGuard - Early Detection of Sleep Disorders")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Summary
        report_lines.append("RISK ASSESSMENT SUMMARY")
        report_lines.append("-"*80)
        report_lines.append(f"Overall Risk Level: {risk_report['risk_level']}")
        report_lines.append(f"Risk Score: {risk_report['risk_score']}/100")
        report_lines.append(f"Analysis Duration: {risk_report['duration_hours']:.2f} hours ({risk_report['total_epochs']} epochs)")
        report_lines.append("")
        
        # Risk Factors
        if risk_report['risk_factors']:
            report_lines.append("IDENTIFIED RISK FACTORS")
            report_lines.append("-"*80)
            for i, factor in enumerate(risk_report['risk_factors'], 1):
                report_lines.append(f"\n{i}. {factor['factor']} [{factor['severity']}] - {factor['points']} points")
                report_lines.append(f"   Value: {factor['value']}")
                report_lines.append(f"   {factor['description']}")
        else:
            report_lines.append("No significant risk factors detected")
        
        report_lines.append("\n")
        report_lines.append("="*80)
        report_lines.append("DETAILED SLEEP ANALYSIS")
        report_lines.append("="*80)
        
        # Stage Distribution
        report_lines.append("\n1. SLEEP STAGE DISTRIBUTION")
        report_lines.append("-"*80)
        dist = risk_report['stage_distribution']
        report_lines.append(f"Wake:       {dist['wake_pct']:6.2f}%")
        report_lines.append(f"N1 (Light): {dist['n1_pct']:6.2f}%")
        report_lines.append(f"N2 (Light): {dist['n2_pct']:6.2f}%")
        report_lines.append(f"N3 (Deep):  {dist['n3_pct']:6.2f}%")
        if dist['n4_pct'] > 0:
            report_lines.append(f"N4 (Deep):  {dist['n4_pct']:6.2f}%")
        report_lines.append(f"REM:        {dist['rem_pct']:6.2f}%")
        report_lines.append(f"Total Deep: {dist['deep_sleep_pct']:6.2f}%")
        
        # REM Analysis
        report_lines.append("\n2. REM SLEEP ANALYSIS")
        report_lines.append("-"*80)
        rem = risk_report['rem_analysis']
        report_lines.append(f"REM Periods: {rem['rem_period_count']}")
        if rem['rem_latency_minutes']:
            report_lines.append(f"REM Latency: {rem['rem_latency_minutes']:.1f} minutes")
            if rem['rem_period_count'] > 0:
                report_lines.append(f"Average REM Period: {rem['avg_rem_period_length']:.1f} minutes")
                report_lines.append(f"Longest REM Period: {rem['longest_rem_period']:.1f} minutes")
                report_lines.append(f"Shortest REM Period: {rem['shortest_rem_period']:.1f} minutes")
        
        # Transition Analysis
        report_lines.append("\n3. SLEEP STAGE TRANSITIONS")
        report_lines.append("-"*80)
        trans = risk_report['transition_analysis']
        report_lines.append(f"Total Transitions: {trans['total_transitions']}")
        report_lines.append(f"Transition Rate: {trans['transition_rate_per_hour']:.1f} per hour")
        report_lines.append(f"REM to Wake: {trans['rem_to_wake_count']} (CRITICAL)")
        report_lines.append(f"Wake to REM: {trans['wake_to_rem_count']}")
        report_lines.append(f"All REM Exits: {trans['rem_to_any_count']}")
        
        if trans['critical_transitions']:
            report_lines.append("\nCritical REM to Wake Transitions:")
            for ct in trans['critical_transitions'][:10]:  # Show first 10
                report_lines.append(f"  - Epoch {ct['epoch']} ({ct['time_hours']:.2f} hours)")
        
        # Fragmentation
        report_lines.append("\n4. SLEEP FRAGMENTATION")
        report_lines.append("-"*80)
        frag = risk_report['fragmentation_analysis']
        report_lines.append(f"Awakenings: {frag['awakening_count']}")
        report_lines.append(f"Awakening Index: {frag['awakening_index']:.1f} per hour")
        report_lines.append(f"Sleep Efficiency: {frag['sleep_efficiency']:.1f}%")
        report_lines.append(f"Wake After Sleep Onset: {frag['wake_after_sleep_onset_minutes']:.1f} minutes")
        
        # Sleep Architecture
        report_lines.append("\n5. SLEEP ARCHITECTURE")
        report_lines.append("-"*80)
        arch = risk_report['sleep_architecture']
        report_lines.append(f"Sleep Cycles Detected: {arch['sleep_cycles_detected']}")
        if arch['notes']:
            for note in arch['notes']:
                report_lines.append(f"  - {note}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        # Save report with UTF-8 encoding (fixes Unicode error)
        report_text = "\n".join(report_lines)
        report_path = os.path.join(output_dir, 'sleep_paralysis_risk_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Also print to console
        print(report_text)
        print(f"\nReport saved to: {os.path.abspath(report_path)}")
        
        return report_path
    
    def visualize_analysis(self, risk_report, sleep_stages, output_dir='./outputs'):
        """
        Generate visualizations of the risk analysis
        
        Args:
            risk_report: Risk analysis results
            sleep_stages: Original sleep stage array
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Hypnogram with Risk Markers
        ax1 = fig.add_subplot(gs[0, :])
        time_hours = np.arange(len(sleep_stages)) * (self.epoch_duration / 3600)
        ax1.plot(time_hours, sleep_stages, linewidth=2, color='#2c3e50', alpha=0.7)
        ax1.fill_between(time_hours, sleep_stages, alpha=0.3, color='#3498db')
        
        # Mark critical REM to Wake transitions
        for trans in risk_report['transition_analysis']['critical_transitions']:
            ax1.axvline(x=trans['time_hours'], color='red', linestyle='--', 
                       alpha=0.5, linewidth=1.5)
        
        unique_stages = sorted(np.unique(sleep_stages))
        stage_names = {-1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N4', 5: 'REM'}
        ax1.set_yticks(unique_stages)
        ax1.set_yticklabels([stage_names.get(int(s), f'S{s}') for s in unique_stages])
        ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax1.set_title('Hypnogram with REM to Wake Transitions (Red Lines)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sleep Stage Distribution Pie Chart
        ax2 = fig.add_subplot(gs[1, 0])
        dist = risk_report['stage_distribution']
        stages_pct = [dist['wake_pct'], dist['n1_pct'], dist['n2_pct'], 
                     dist['deep_sleep_pct'], dist['rem_pct']]
        labels = ['Wake', 'N1', 'N2', 'N3/N4', 'REM']
        colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']
        
        ax2.pie(stages_pct, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90)
        ax2.set_title('Sleep Stage Distribution', fontsize=12, fontweight='bold')
        
        # 3. Risk Factor Contributions
        ax3 = fig.add_subplot(gs[1, 1])
        if risk_report['risk_factors']:
            factors = [rf['factor'] for rf in risk_report['risk_factors']]
            points = [rf['points'] for rf in risk_report['risk_factors']]
            colors_risk = ['#e74c3c' if rf['severity'] == 'HIGH' else 
                          '#f39c12' if rf['severity'] == 'MODERATE' else '#3498db' 
                          for rf in risk_report['risk_factors']]
            
            ax3.barh(factors, points, color=colors_risk)
            ax3.set_xlabel('Risk Points', fontsize=11)
            ax3.set_title('Risk Factor Contributions', fontsize=12, fontweight='bold')
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, 'No Risk Factors Detected', 
                    ha='center', va='center', fontsize=14)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
        
        # 4. Transition Matrix Heatmap (Top 10)
        ax4 = fig.add_subplot(gs[2, 0])
        trans_matrix = risk_report['transition_analysis']['transition_matrix']
        sorted_trans = sorted(trans_matrix.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_trans:
            trans_names = [t[0] for t in sorted_trans]
            trans_counts = [t[1] for t in sorted_trans]
            
            ax4.barh(range(len(trans_names)), trans_counts, color='#3498db')
            ax4.set_yticks(range(len(trans_names)))
            ax4.set_yticklabels(trans_names, fontsize=9)
            ax4.set_xlabel('Count', fontsize=11)
            ax4.set_title('Top 10 Sleep Stage Transitions', fontsize=12, fontweight='bold')
            ax4.invert_yaxis()
        
        # 5. Risk Score Gauge
        ax5 = fig.add_subplot(gs[2, 1])
        risk_score = risk_report['risk_score']
        risk_level = risk_report['risk_level']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones(100)
        
        # Background arc
        ax5.fill_between(theta, 0, r, where=(theta <= np.pi/3), 
                        color='#2ecc71', alpha=0.3, label='Low Risk (0-25)')
        ax5.fill_between(theta, 0, r, where=((theta > np.pi/3) & (theta <= 2*np.pi/3)), 
                        color='#f39c12', alpha=0.3, label='Moderate (25-50)')
        ax5.fill_between(theta, 0, r, where=(theta > 2*np.pi/3), 
                        color='#e74c3c', alpha=0.3, label='High Risk (50+)')
        
        # Score needle
        score_angle = np.pi * (1 - risk_score/100)
        ax5.plot([0, np.cos(score_angle)], [0, np.sin(score_angle)], 
                'k-', linewidth=3)
        ax5.plot(0, 0, 'ko', markersize=10)
        
        # Labels
        ax5.text(0, -0.3, f'{risk_score}', ha='center', fontsize=24, fontweight='bold')
        ax5.text(0, -0.5, risk_level, ha='center', fontsize=16, 
                color='#e74c3c' if risk_level=='HIGH' else 
                      '#f39c12' if risk_level=='MODERATE' else '#2ecc71')
        
        ax5.set_xlim(-1.2, 1.2)
        ax5.set_ylim(-0.6, 1.2)
        ax5.axis('off')
        ax5.set_title('Sleep Paralysis Risk Score', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=8)
        
        # Overall title
        fig.suptitle('Sleep Paralysis Risk Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        viz_path = os.path.join(output_dir, 'sleep_paralysis_risk_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {os.path.abspath(viz_path)}")
        
        return fig


def main():
    """
    Main execution - Load predictions from Step 1 and analyze
    """
    print("="*80)
    print("STEP 2: SLEEP PARALYSIS RISK ANALYSIS")
    print("SomnusGuard - Analyzing Hypnogram from Step 1")
    print("="*80)
    
    # ========================================================================
    # LOAD PREDICTIONS FROM STEP 1
    # ========================================================================
    
    predictions_file = './outputs/predictions.npy'
    
    # Check if Step 1 has been run
    if not os.path.exists(predictions_file):
        print("\n" + "="*80)
        print("ERROR: predictions.npy not found!")
        print("="*80)
        print("\nYou need to run Step 1 first:")
        print("  1. Run: python sleep_paralysis_rf_classifier.py")
        print("  2. Wait for it to complete and save predictions.npy")
        print("  3. Then run this script again")
        print("\nExpected file location:")
        print(f"  {os.path.abspath(predictions_file)}")
        print("="*80)
        return
    
    # Load predictions (hypnogram)
    print(f"\nLoading predictions from Step 1...")
    print(f"File: {os.path.abspath(predictions_file)}")
    predictions = np.load(predictions_file)
    
    print(f"✓ Loaded hypnogram with {len(predictions)} epochs")
    print(f"✓ Duration: {len(predictions)*0.5/60:.2f} hours")
    
    # Show what we loaded
    print("\nSleep stage distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    stage_names = {-1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 
                   3: 'N3', 4: 'N4', 5: 'REM'}
    
    for stage, count in zip(unique, counts):
        pct = (count / len(predictions)) * 100
        stage_name = stage_names.get(int(stage), f'Stage_{stage}')
        print(f"  {stage_name:8s}: {count:5d} epochs ({pct:5.1f}%)")
    
    # ========================================================================
    # ANALYZE SLEEP PARALYSIS RISK
    # ========================================================================
    
    print("\n" + "="*80)
    print("ANALYZING SLEEP PATTERNS FOR PARALYSIS RISK")
    print("="*80)
    
    # Initialize analyzer
    analyzer = SleepParalysisRiskAnalyzer(epoch_duration=30)
    
    # Analyze the hypnogram
    print("\nRunning risk analysis...")
    risk_report = analyzer.analyze_hypnogram(predictions)
    
    # Generate outputs
    print("\nGenerating reports...")
    analyzer.generate_report(risk_report, output_dir='./outputs')
    analyzer.visualize_analysis(risk_report, predictions, output_dir='./outputs')
    
    # ========================================================================
    # DISPLAY RESULTS SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2 ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\n🎯 SLEEP PARALYSIS RISK ASSESSMENT:")
    print(f"   Overall Risk Level: {risk_report['risk_level']}")
    print(f"   Risk Score: {risk_report['risk_score']}/100")
    
    if risk_report['risk_factors']:
        print(f"\n⚠️  RISK FACTORS IDENTIFIED: {len(risk_report['risk_factors'])}")
        for i, factor in enumerate(risk_report['risk_factors'], 1):
            print(f"\n   {i}. {factor['factor']} [{factor['severity']}]")
            print(f"      Points: {factor['points']}")
            print(f"      Value: {factor['value']}")
    else:
        print("\n✓ No significant risk factors detected")
        print("✓ Sleep patterns appear normal")
    
    # Key metrics
    print(f"\n📊 KEY SLEEP METRICS:")
    print(f"   REM Sleep: {risk_report['stage_distribution']['rem_pct']:.1f}%")
    print(f"   Deep Sleep: {risk_report['stage_distribution']['deep_sleep_pct']:.1f}%")
    print(f"   REM Periods: {risk_report['rem_analysis']['rem_period_count']}")
    print(f"   REM to Wake Transitions: {risk_report['transition_analysis']['rem_to_wake_count']} (CRITICAL)")
    print(f"   Sleep Efficiency: {risk_report['fragmentation_analysis']['sleep_efficiency']:.1f}%")
    
    print(f"\n📁 OUTPUT FILES GENERATED:")
    print("   ✓ sleep_paralysis_risk_report.txt")
    print("   ✓ sleep_paralysis_risk_visualization.png")
    
    print("\n" + "="*80)
    print("✅ ALL DONE! Check the ./outputs folder for detailed results.")
    print("="*80)


if __name__ == "__main__":
    main()