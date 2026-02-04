"""
Gate Definitions for Axiom Nexus
Loads gate configuration from gates.yaml
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Gate:
    """Represents a single knowledge gate"""
    id: int
    name: str
    display_name: str
    description: str
    specialist_model: str
    provider: str
    context_window: int
    reasoning: str
    keywords: List[str]
    verifier: str
    verification_strict: bool
    
    def __hash__(self):
        return hash(self.name)
    
    def matches_keyword(self, text: str) -> bool:
        """Check if text contains any gate keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.keywords)


class GateConfig:
    """Manages gate configuration from YAML"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "gates.yaml"
        
        self.config_path = config_path
        self.gates: Dict[str, Gate] = {}
        self.gates_by_id: Dict[int, Gate] = {}
        self._load_config()
    
    def _load_config(self):
        """Load gates from YAML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for gate_data in config.get('gates', []):
                gate = Gate(
                    id=gate_data['id'],
                    name=gate_data['name'],
                    display_name=gate_data['display_name'],
                    description=gate_data['description'],
                    specialist_model=gate_data['specialist_model'],
                    provider=gate_data['provider'],
                    context_window=gate_data['context_window'],
                    reasoning=gate_data['reasoning'],
                    keywords=gate_data['keywords'],
                    verifier=gate_data['verifier'],
                    verification_strict=gate_data['verification_strict']
                )
                self.gates[gate.name] = gate
                self.gates_by_id[gate.id] = gate
            
            logger.info(f"Loaded {len(self.gates)} gates from configuration")
            
        except Exception as e:
            logger.error(f"Failed to load gate configuration: {e}")
            raise
    
    def get_gate(self, name: str) -> Optional[Gate]:
        """Get gate by name"""
        return self.gates.get(name)
    
    def get_gate_by_id(self, gate_id: int) -> Optional[Gate]:
        """Get gate by ID"""
        return self.gates_by_id.get(gate_id)
    
    def get_all_gates(self) -> List[Gate]:
        """Get all gates"""
        return list(self.gates.values())


# Global gate configuration instance
_gate_config = None

def get_gate_config() -> GateConfig:
    """Get or create gate configuration singleton"""
    global _gate_config
    if _gate_config is None:
        _gate_config = GateConfig()
    return _gate_config


# Convenience functions
def get_gate_by_name(name: str) -> Optional[Gate]:
    """Get gate by name"""
    return get_gate_config().get_gate(name)


def get_gate_by_id(gate_id: int) -> Optional[Gate]:
    """Get gate by ID"""
    return get_gate_config().get_gate_by_id(gate_id)


def get_all_gates() -> List[Gate]:
    """Get all gates"""
    return get_gate_config().get_all_gates()
