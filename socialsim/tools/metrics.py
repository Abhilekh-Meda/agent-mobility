"""
Metrics collection and logging system for SocialSim.

Tracks simulation metrics and exports to CSV/JSON.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from socialsim.core.types import StepMetrics


class MetricsCollector:
    """Collects and stores simulation metrics.
    
    Features:
    - Per-step metrics tracking
    - CSV export
    - JSON export
    - Summary statistics
    - Custom metrics support
    """
    
    def __init__(
        self,
        simulation_name: str,
        output_dir: str = "./logs"
    ):
        """Initialize metrics collector.
        
        Args:
            simulation_name: Name of simulation
            output_dir: Directory for output files
        """
        self.simulation_name = simulation_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.step_metrics: List[StepMetrics] = []
        self.custom_metrics: Dict[str, List[Any]] = {}
        
        # Metadata
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        logger.debug(f"Initialized MetricsCollector for '{simulation_name}'")
    
    def record_step(self, metrics: StepMetrics) -> None:
        """Record metrics for a simulation step.
        
        Args:
            metrics: StepMetrics object
        """
        self.step_metrics.append(metrics)
    
    def record_custom(self, metric_name: str, value: Any) -> None:
        """Record a custom metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = []
        
        self.custom_metrics[metric_name].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Dictionary with summary stats
        """
        if not self.step_metrics:
            return {"error": "No metrics recorded"}
        
        # Calculate aggregates
        total_steps = len(self.step_metrics)
        total_actions = sum(m.actions_taken for m in self.step_metrics)
        total_llm_calls = sum(m.llm_calls for m in self.step_metrics)
        total_duration = sum(m.step_duration_seconds for m in self.step_metrics)
        
        avg_agents = sum(m.num_agents for m in self.step_metrics) / total_steps
        avg_actions_per_step = total_actions / total_steps
        avg_step_duration = total_duration / total_steps
        
        return {
            "simulation_name": self.simulation_name,
            "total_steps": total_steps,
            "total_actions": total_actions,
            "total_llm_calls": total_llm_calls,
            "total_duration_seconds": total_duration,
            "average_agents": avg_agents,
            "average_actions_per_step": avg_actions_per_step,
            "average_step_duration_seconds": avg_step_duration,
            "steps_per_second": total_steps / max(0.001, total_duration),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }
    
    def save_csv(self, filepath: Optional[str] = None) -> str:
        """Save metrics to CSV file.
        
        Args:
            filepath: Output file path (auto-generated if None)
            
        Returns:
            Path where file was saved
        """
        if not self.step_metrics:
            logger.warning("No metrics to save")
            return ""
        
        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"{self.simulation_name}_{timestamp}.csv"
        else:
            filepath = Path(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            # Get all possible fields
            fieldnames = [
                "step",
                "timestamp",
                "num_agents",
                "actions_taken",
                "llm_calls",
                "step_duration_seconds"
            ]
            
            # Add custom metric fields
            if self.step_metrics[0].custom_metrics:
                custom_fields = list(self.step_metrics[0].custom_metrics.keys())
                fieldnames.extend(custom_fields)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write each step
            for metrics in self.step_metrics:
                row = {
                    "step": metrics.step,
                    "timestamp": metrics.timestamp.isoformat(),
                    "num_agents": metrics.num_agents,
                    "actions_taken": metrics.actions_taken,
                    "llm_calls": metrics.llm_calls,
                    "step_duration_seconds": metrics.step_duration_seconds
                }
                
                # Add custom metrics
                row.update(metrics.custom_metrics)
                
                writer.writerow(row)
        
        logger.info(f"Saved metrics to {filepath}")
        return str(filepath)
    
    def save_json(self, filepath: Optional[str] = None) -> str:
        """Save metrics to JSON file.
        
        Args:
            filepath: Output file path (auto-generated if None)
            
        Returns:
            Path where file was saved
        """
        if not self.step_metrics:
            logger.warning("No metrics to save")
            return ""
        
        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"{self.simulation_name}_{timestamp}.json"
        else:
            filepath = Path(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = {
            "simulation_name": self.simulation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "summary": self.get_summary(),
            "steps": [
                {
                    "step": m.step,
                    "timestamp": m.timestamp.isoformat(),
                    "num_agents": m.num_agents,
                    "actions_taken": m.actions_taken,
                    "llm_calls": m.llm_calls,
                    "step_duration_seconds": m.step_duration_seconds,
                    "custom_metrics": m.custom_metrics
                }
                for m in self.step_metrics
            ],
            "custom_metrics": self.custom_metrics
        }
        
        # Write JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")
        return str(filepath)
    
    def save(self, filepath: Optional[str] = None, format: str = "csv") -> str:
        """Save metrics to file.
        
        Args:
            filepath: Output file path
            format: File format ('csv' or 'json')
            
        Returns:
            Path where file was saved
        """
        self.end_time = datetime.now()
        
        if format == "csv":
            return self.save_csv(filepath)
        elif format == "json":
            return self.save_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'csv' or 'json'")
    
    def get_step_metrics(self, step: int) -> Optional[StepMetrics]:
        """Get metrics for a specific step.
        
        Args:
            step: Step number
            
        Returns:
            StepMetrics or None if not found
        """
        for metrics in self.step_metrics:
            if metrics.step == step:
                return metrics
        return None
    
    def get_metrics_range(
        self,
        start_step: int,
        end_step: int
    ) -> List[StepMetrics]:
        """Get metrics for a range of steps.
        
        Args:
            start_step: Start step (inclusive)
            end_step: End step (inclusive)
            
        Returns:
            List of StepMetrics in range
        """
        return [
            m for m in self.step_metrics
            if start_step <= m.step <= end_step
        ]
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot key metrics over time (requires matplotlib).
        
        Args:
            save_path: Path to save plot image (if None, display only)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if not self.step_metrics:
            logger.warning("No metrics to plot")
            return
        
        # Extract data
        steps = [m.step for m in self.step_metrics]
        actions = [m.actions_taken for m in self.step_metrics]
        llm_calls = [m.llm_calls for m in self.step_metrics]
        durations = [m.step_duration_seconds for m in self.step_metrics]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"Simulation Metrics: {self.simulation_name}", fontsize=16)
        
        # Plot actions
        axes[0].plot(steps, actions, 'b-', linewidth=2)
        axes[0].set_ylabel('Actions Taken')
        axes[0].set_title('Actions per Step')
        axes[0].grid(True, alpha=0.3)
        
        # Plot LLM calls
        axes[1].plot(steps, llm_calls, 'r-', linewidth=2)
        axes[1].set_ylabel('LLM Calls')
        axes[1].set_title('LLM API Calls per Step')
        axes[1].grid(True, alpha=0.3)
        
        # Plot step duration
        axes[2].plot(steps, durations, 'g-', linewidth=2)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Duration (seconds)')
        axes[2].set_title('Step Execution Time')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        self.step_metrics.clear()
        self.custom_metrics.clear()
        self.start_time = datetime.now()
        self.end_time = None
        logger.debug("Metrics cleared")
    
    def __len__(self) -> int:
        return len(self.step_metrics)
    
    def __str__(self) -> str:
        return f"MetricsCollector('{self.simulation_name}', {len(self.step_metrics)} steps)"
    
    def __repr__(self) -> str:
        return f"<MetricsCollector for '{self.simulation_name}'>"