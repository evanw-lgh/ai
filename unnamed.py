
#This is a pygame hide and seek thingy

import pygame
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WORLD_WIDTH = 800
WORLD_HEIGHT = 600
PANEL_WIDTH = WINDOW_WIDTH - WORLD_WIDTH

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 200, 50)
DARK_GREEN = (30, 150, 30)
RED = (255, 50, 50)
BLUE = (50, 100, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (80, 80, 80)
YELLOW = (255, 255, 0)
PURPLE = (150, 50, 200)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
BROWN = (139, 69, 19)

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    movable: bool = False
    being_pushed: bool = False
    push_direction: Tuple[float, float] = (0, 0)

@dataclass
class Agent:
    x: float
    y: float
    speed: float
    vision_range: float
    memory: List[Tuple[float, float]]
    fitness: float = 0
    alive: bool = True
    trail: List[Tuple[int, int]] = None
    pushing_obstacle: bool = False
    target_obstacle: Obstacle = None
    cooperation_memory: List[Tuple[float, float]] = None
    last_seen_enemies: List[Tuple[float, float, int]] = None  # x, y, timestamp
    strategic_state: str = "exploring"  # exploring, hiding, hunting, cooperating
    energy: float = 100.0

    def __post_init__(self):
        self.memory = self.memory or []
        self.trail = self.trail or []
        self.cooperation_memory = self.cooperation_memory or []
        self.last_seen_enemies = self.last_seen_enemies or []


class PyGameHideSeek:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Enhanced AI Hide and Seek - Genetic Evolution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Game parameters
        self.world_width = WORLD_WIDTH
        self.world_height = WORLD_HEIGHT
        self.n_hiders = 6
        self.n_seekers = 3
        self.obstacles = self.generate_obstacles()

        # Evolution parameters
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.generation = 0
        self.game_time = 0
        self.max_game_time = 900  # 15 seconds at 60 FPS

        # Populations (expanded gene set)
        self.hider_population = self.create_initial_population('hider')
        self.seeker_population = self.create_initial_population('seeker')

        # Current game agents
        self.hiders = []
        self.seekers = []
        self.start_new_game()

        # Statistics
        self.hider_fitness_history = []
        self.seeker_fitness_history = []
        self.survival_rates = []
        self.generation_stats = []

        # Display options
        self.show_vision = True
        self.show_trails = True
        self.show_genes = True
        self.show_states = True
        self.speed_multiplier = 1
        self.paused = False

    def generate_obstacles(self):
        """Generate random obstacles with some movable blocks"""
        obstacles = []
        
        # Fixed large obstacles
        for _ in range(random.randint(4, 7)):
            x = random.uniform(80, self.world_width - 80)
            y = random.uniform(80, self.world_height - 80)
            radius = random.uniform(25, 45)
            obstacles.append(Obstacle(x, y, radius, movable=False))
        
        # Movable blocks
        for _ in range(random.randint(6, 12)):
            x = random.uniform(60, self.world_width - 60)
            y = random.uniform(60, self.world_height - 60)
            radius = random.uniform(15, 25)
            obstacles.append(Obstacle(x, y, radius, movable=True))
        
        return obstacles

    def create_initial_population(self, agent_type):
        """Create initial population with expanded gene set"""
        population = []
        count = self.n_hiders if agent_type == 'hider' else self.n_seekers

        for _ in range(count):
            genes = [
                random.uniform(1.0, 4.5),    # Speed
                random.uniform(50, 140),     # Vision range
                random.uniform(0, 1),        # Aggression/hiding tendency
                random.uniform(0, 1),        # Wall preference
                random.uniform(0, 1),        # Memory importance
                random.uniform(0, 1),        # Block manipulation skill
                random.uniform(0, 1),        # Cooperation tendency
                random.uniform(0, 1),        # Strategic thinking
                random.uniform(0, 1),        # Energy efficiency
                random.uniform(0, 1)         # Prediction ability
            ]
            population.append(genes)

        return population

    def distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def create_agent_from_genes(self, genes, agent_type):
        """Create agent from genetic encoding"""
        attempts = 0
        while attempts < 50:
            x = random.uniform(30, self.world_width - 30)
            y = random.uniform(30, self.world_height - 30)

            # Check if position is valid
            valid = True
            for obstacle in self.obstacles:
                if self.distance((x, y), (obstacle.x, obstacle.y)) < obstacle.radius + 15:
                    valid = False
                    break

            if valid:
                break
            attempts += 1

        return Agent(
            x=x, y=y,
            speed=genes[0],
            vision_range=genes[1],
            memory=[],
            trail=[],
            energy=100.0
        )

    def start_new_game(self):
        """Start a new game with current populations"""
        self.hiders = [self.create_agent_from_genes(genes, 'hider')
                       for genes in self.hider_population]
        self.seekers = [self.create_agent_from_genes(genes, 'seeker')
                        for genes in self.seeker_population]
        self.game_time = 0
        
        # Reset obstacle states
        for obstacle in self.obstacles:
            obstacle.being_pushed = False
            obstacle.push_direction = (0, 0)

    def can_push_obstacle(self, agent, obstacle, genes):
        """Check if agent can push an obstacle"""
        if not obstacle.movable:
            return False
        
        block_skill = genes[5]  # Block manipulation skill
        distance = self.distance((agent.x, agent.y), (obstacle.x, obstacle.y))
        
        return distance < obstacle.radius + 20 and block_skill > 0.3

    def push_obstacle(self, agent, obstacle, direction, genes):
        """Push an obstacle in a direction"""
        if not self.can_push_obstacle(agent, obstacle, genes):
            return False
        
        block_skill = genes[5]
        push_strength = block_skill * 0.8
        
        # Calculate new obstacle position
        new_x = obstacle.x + direction[0] * push_strength
        new_y = obstacle.y + direction[1] * push_strength
        
        # Boundary checking
        new_x = max(obstacle.radius, min(self.world_width - obstacle.radius, new_x))
        new_y = max(obstacle.radius, min(self.world_height - obstacle.radius, new_y))
        
        # Check collision with other obstacles
        valid_move = True
        for other_obstacle in self.obstacles:
            if other_obstacle != obstacle:
                if self.distance((new_x, new_y), (other_obstacle.x, other_obstacle.y)) < obstacle.radius + other_obstacle.radius + 5:
                    valid_move = False
                    break
        
        if valid_move:
            obstacle.x, obstacle.y = new_x, new_y
            obstacle.being_pushed = True
            obstacle.push_direction = direction
            agent.pushing_obstacle = True
            agent.energy -= 2  # Pushing costs energy
            return True
        
        return False

    def find_strategic_position(self, agent, genes, agent_type, other_agents):
        """Find strategic position based on AI analysis"""
        strategic_thinking = genes[7]
        prediction_ability = genes[9]
        
        if strategic_thinking < 0.4:
            return None
        
        best_position = None
        best_score = -float('inf')
        
        # Sample potential positions
        for _ in range(int(strategic_thinking * 20)):
            test_x = random.uniform(30, self.world_width - 30)
            test_y = random.uniform(30, self.world_height - 30)
            
            # Skip if position is blocked
            blocked = False
            for obstacle in self.obstacles:
                if self.distance((test_x, test_y), (obstacle.x, obstacle.y)) < obstacle.radius + 15:
                    blocked = True
                    break
            
            if blocked:
                continue
            
            score = self.evaluate_position(test_x, test_y, agent, genes, agent_type, other_agents)
            
            if score > best_score:
                best_score = score
                best_position = (test_x, test_y)
        
        return best_position

    def evaluate_position(self, x, y, agent, genes, agent_type, other_agents):
        """Evaluate how good a position is strategically"""
        score = 0
        wall_pref = genes[3]
        
        # Distance to walls (good for hiders)
        wall_distances = [x, y, self.world_width - x, self.world_height - y]
        min_wall_dist = min(wall_distances)
        
        if agent_type == 'hider':
            # Hiders prefer positions near walls and obstacles
            score += (100 - min_wall_dist) * wall_pref * 0.01
            
            # Distance from seekers (farther is better)
            for seeker in other_agents:
                dist = self.distance((x, y), (seeker.x, seeker.y))
                score += min(dist * 0.02, 5)
            
            # Near cover (obstacles)
            for obstacle in self.obstacles:
                dist = self.distance((x, y), (obstacle.x, obstacle.y))
                if dist < obstacle.radius + 40:
                    score += 3
        
        else:  # seeker
            # Seekers prefer central positions with good visibility
            center_dist = self.distance((x, y), (self.world_width/2, self.world_height/2))
            score += (200 - center_dist) * 0.01
            
            # Near hiders (closer is better)
            for hider in other_agents:
                if hider.alive:
                    dist = self.distance((x, y), (hider.x, hider.y))
                    score += max(0, 10 - dist * 0.1)
        
        return score

    def update_agent_state(self, agent, genes, agent_type, other_agents):
        """Update agent's strategic state"""
        cooperation = genes[6]
        strategic_thinking = genes[7]
        
        if strategic_thinking < 0.3:
            agent.strategic_state = "exploring"
            return
        
        visible_enemies = []
        for other in other_agents:
            if (agent_type == 'hider' and not other.alive) or (agent_type == 'seeker' and other.alive):
                continue
            if self.distance((agent.x, agent.y), (other.x, other.y)) <= agent.vision_range:
                visible_enemies.append(other)
        
        if agent_type == 'hider':
            if visible_enemies:
                agent.strategic_state = "hiding"
            elif cooperation > 0.6 and len([h for h in other_agents if h.alive]) > 2:
                agent.strategic_state = "cooperating"
            else:
                agent.strategic_state = "exploring"
        else:  # seeker
            if visible_enemies:
                agent.strategic_state = "hunting"
            elif cooperation > 0.6:
                agent.strategic_state = "cooperating"
            else:
                agent.strategic_state = "exploring"

    def move_agent(self, agent, genes, agent_type, other_agents):
        """Enhanced agent movement with block manipulation"""
        if not agent.alive:
            return

        speed, vision_range, aggression, wall_pref, memory_weight = genes[:5]
        block_skill, cooperation, strategic_thinking, energy_efficiency, prediction = genes[5:]

        # Update strategic state
        self.update_agent_state(agent, genes, agent_type, other_agents)
        
        # Energy management
        if energy_efficiency > 0.5:
            agent.energy = min(100, agent.energy + 0.1)
        
        if agent.energy < 20:
            speed *= 0.5  # Slow down when tired

        # Find visible agents and obstacles
        visible_agents = []
        visible_obstacles = []
        
        for other in other_agents:
            if other.alive and self.distance((agent.x, agent.y), (other.x, other.y)) <= vision_range:
                visible_agents.append(other)
        
        for obstacle in self.obstacles:
            if self.distance((agent.x, agent.y), (obstacle.x, obstacle.y)) <= vision_range:
                visible_obstacles.append(obstacle)

        dx, dy = 0, 0
        
        # Strategic positioning
        if strategic_thinking > 0.6 and random.random() < 0.1:
            strategic_pos = self.find_strategic_position(agent, genes, agent_type, other_agents)
            if strategic_pos:
                target_dx = strategic_pos[0] - agent.x
                target_dy = strategic_pos[1] - agent.y
                target_dist = math.sqrt(target_dx ** 2 + target_dy ** 2)
                if target_dist > 20:
                    dx += (target_dx / target_dist) * strategic_thinking * 0.3
                    dy += (target_dy / target_dist) * strategic_thinking * 0.3

        if agent_type == 'hider':
            # Enhanced hider behavior
            if agent.strategic_state == "hiding":
                # Flee from seekers with prediction
                for seeker in visible_agents:
                    dist = self.distance((agent.x, agent.y), (seeker.x, seeker.y))
                    if dist > 0:
                        # Predict seeker movement
                        predicted_x = seeker.x
                        predicted_y = seeker.y
                        if prediction > 0.5 and len(seeker.trail) > 2:
                            last_pos = seeker.trail[-1]
                            second_last = seeker.trail[-2]
                            predicted_x += (last_pos[0] - second_last[0]) * prediction * 3
                            predicted_y += (last_pos[1] - second_last[1]) * prediction * 3
                        
                        flee_strength = max(0.8, 3.0 - dist / vision_range)
                        dx += (agent.x - predicted_x) / dist * flee_strength
                        dy += (agent.y - predicted_y) / dist * flee_strength

            elif agent.strategic_state == "cooperating":
                # Group with other hiders
                for other_hider in [h for h in other_agents if h.alive and h != agent]:
                    dist = self.distance((agent.x, agent.y), (other_hider.x, other_hider.y))
                    if 30 < dist < 80:
                        dx += (other_hider.x - agent.x) / dist * cooperation * 0.3
                        dy += (other_hider.y - agent.y) / dist * cooperation * 0.3

            # Block manipulation for cover
            if block_skill > 0.4 and not agent.pushing_obstacle:
                best_obstacle = None
                best_score = -1
                
                for obstacle in visible_obstacles:
                    if obstacle.movable and not obstacle.being_pushed:
                        # Calculate if moving this obstacle would provide better cover
                        score = 0
                        for seeker in visible_agents:
                            # Check if obstacle could block line of sight
                            seeker_dist = self.distance((agent.x, agent.y), (seeker.x, seeker.y))
                            obstacle_dist = self.distance((agent.x, agent.y), (obstacle.x, obstacle.y))
                            if obstacle_dist < seeker_dist:
                                score += 1
                        
                        if score > best_score and self.can_push_obstacle(agent, obstacle, genes):
                            best_score = score
                            best_obstacle = obstacle
                
                if best_obstacle and best_score > 0:
                    # Push obstacle to create cover
                    push_dir_x = agent.x - best_obstacle.x
                    push_dir_y = agent.y - best_obstacle.y
                    push_magnitude = math.sqrt(push_dir_x**2 + push_dir_y**2)
                    if push_magnitude > 0:
                        push_dir_x /= push_magnitude
                        push_dir_y /= push_magnitude
                        self.push_obstacle(agent, best_obstacle, (-push_dir_x, -push_dir_y), genes)

            # Seek cover behind obstacles
            if wall_pref > 0.4:
                best_cover = None
                best_cover_score = -1
                
                # Check obstacles for cover
                for obstacle in visible_obstacles:
                    cover_score = 0
                    for seeker in visible_agents:
                        # Calculate cover effectiveness
                        to_obstacle = self.distance((agent.x, agent.y), (obstacle.x, obstacle.y))
                        to_seeker = self.distance((agent.x, agent.y), (seeker.x, seeker.y))
                        if to_obstacle < to_seeker:
                            cover_score += obstacle.radius / to_obstacle
                    
                    if cover_score > best_cover_score:
                        best_cover_score = cover_score
                        best_cover = obstacle
                
                if best_cover:
                    # Move to cover position
                    angle = math.atan2(agent.y - best_cover.y, agent.x - best_cover.x)
                    target_x = best_cover.x + (best_cover.radius + 20) * math.cos(angle)
                    target_y = best_cover.y + (best_cover.radius + 20) * math.sin(angle)
                    
                    cover_dx = target_x - agent.x
                    cover_dy = target_y - agent.y
                    cover_dist = math.sqrt(cover_dx**2 + cover_dy**2)
                    if cover_dist > 15:
                        dx += (cover_dx / cover_dist) * wall_pref * 0.8
                        dy += (cover_dy / cover_dist) * wall_pref * 0.8

        else:  # seeker
            if agent.strategic_state == "hunting":
                # Enhanced hunting with prediction
                if visible_agents:
                    nearest = min(visible_agents,
                                  key=lambda h: self.distance((agent.x, agent.y), (h.x, h.y)))
                    dist = self.distance((agent.x, agent.y), (nearest.x, nearest.y))
                    
                    # Predict hider movement
                    target_x, target_y = nearest.x, nearest.y
                    if prediction > 0.5 and len(nearest.trail) > 2:
                        last_pos = nearest.trail[-1]
                        second_last = nearest.trail[-2]
                        target_x += (last_pos[0] - second_last[0]) * prediction * 2
                        target_y += (last_pos[1] - second_last[1]) * prediction * 2
                    
                    if dist > 0:
                        chase_strength = aggression * (2.0 - dist / vision_range)
                        dx += (target_x - agent.x) / dist * chase_strength
                        dy += (target_y - agent.y) / dist * chase_strength

                    # Store in memory
                    agent.memory.append((nearest.x, nearest.y))
                    if len(agent.memory) > 12:
                        agent.memory.pop(0)

            elif agent.strategic_state == "cooperating":
                # Coordinate with other seekers
                for other_seeker in [s for s in other_agents if s != agent]:
                    # Share information and coordinate positions
                    if len(other_seeker.memory) > 0:
                        shared_memory = other_seeker.memory[-1]
                        dist_to_shared = self.distance((agent.x, agent.y), shared_memory)
                        if dist_to_shared > 0:
                            dx += (shared_memory[0] - agent.x) / dist_to_shared * cooperation * 0.4
                            dy += (shared_memory[1] - agent.y) / dist_to_shared * cooperation * 0.4

            # Block manipulation for hunting
            if block_skill > 0.5 and not agent.pushing_obstacle:
                for obstacle in visible_obstacles:
                    if obstacle.movable and not obstacle.being_pushed:
                        for hider in visible_agents:
                            # Check if moving obstacle would expose hider
                            hider_obstacle_dist = self.distance((hider.x, hider.y), (obstacle.x, obstacle.y))
                            if hider_obstacle_dist < obstacle.radius + 30:
                                # Push obstacle away from hider
                                push_dir_x = obstacle.x - hider.x
                                push_dir_y = obstacle.y - hider.y
                                push_magnitude = math.sqrt(push_dir_x**2 + push_dir_y**2)
                                if push_magnitude > 0:
                                    push_dir_x /= push_magnitude
                                    push_dir_y /= push_magnitude
                                    if self.can_push_obstacle(agent, obstacle, genes):
                                        self.push_obstacle(agent, obstacle, (push_dir_x, push_dir_y), genes)
                                        break

            # Memory-based search
            if len(agent.memory) > 0 and memory_weight > 0.4 and not visible_agents:
                last_pos = agent.memory[-1]
                dist = self.distance((agent.x, agent.y), last_pos)
                if dist > 0:
                    dx += (last_pos[0] - agent.x) / dist * memory_weight * 0.6
                    dy += (last_pos[1] - agent.y) / dist * memory_weight * 0.6
            elif not visible_agents:
                # Intelligent search pattern
                search_angle = (self.game_time * 0.03 + hash(id(agent)) % 100) % (2 * math.pi)
                search_radius = 50 + strategic_thinking * 100
                center_x = self.world_width // 2 + search_radius * math.cos(search_angle)
                center_y = self.world_height // 2 + search_radius * math.sin(search_angle)
                
                search_dx = center_x - agent.x
                search_dy = center_y - agent.y
                search_dist = math.sqrt(search_dx**2 + search_dy**2)
                if search_dist > 0:
                    dx += (search_dx / search_dist) * 0.5
                    dy += (search_dy / search_dist) * 0.5

        # Add exploration randomness (reduced when tired)
        exploration_factor = 0.4 if agent.energy > 50 else 0.2
        dx += random.uniform(-exploration_factor, exploration_factor)
        dy += random.uniform(-exploration_factor, exploration_factor)

        # Normalize and apply speed
        magnitude = math.sqrt(dx * dx + dy * dy)
        if magnitude > 0:
            dx = (dx / magnitude) * speed * self.speed_multiplier
            dy = (dy / magnitude) * speed * self.speed_multiplier

        # Calculate new position
        new_x = agent.x + dx
        new_y = agent.y + dy

        # Boundary checking
        new_x = max(15, min(self.world_width - 15, new_x))
        new_y = max(15, min(self.world_height - 15, new_y))

        # Obstacle collision
        valid_move = True
        for obstacle in self.obstacles:
            if self.distance((new_x, new_y), (obstacle.x, obstacle.y)) < obstacle.radius + 12:
                valid_move = False
                break

        if valid_move:
            agent.x, agent.y = new_x, new_y
            agent.energy -= 0.1  # Movement costs energy

            # Update trail
            if self.show_trails:
                agent.trail.append((int(agent.x), int(agent.y)))
                if len(agent.trail) > 40:
                    agent.trail.pop(0)

        # Reset pushing state
        agent.pushing_obstacle = False

    def update_game(self):
        """Update one frame of the game"""
        if self.paused:
            return

        self.game_time += 1

        # Reset obstacle pushing states
        for obstacle in self.obstacles:
            obstacle.being_pushed = False

        # Move agents
        for i, hider in enumerate(self.hiders):
            if hider.alive:
                self.move_agent(hider, self.hider_population[i], 'hider', self.seekers)

        for i, seeker in enumerate(self.seekers):
            self.move_agent(seeker, self.seeker_population[i], 'seeker', self.hiders)

        # Check captures
        for hider in self.hiders:
            if hider.alive:
                for seeker in self.seekers:
                    if self.distance((hider.x, hider.y), (seeker.x, seeker.y)) < 28:
                        hider.alive = False
                        break

        # End game condition
        if self.game_time >= self.max_game_time or not any(h.alive for h in self.hiders):
            self.end_game()

    def end_game(self):
        """End current game and evolve"""
        survivors = sum(1 for h in self.hiders if h.alive)
        survival_rate = survivors / len(self.hiders)

        # Enhanced fitness calculation
        hider_fitness = survival_rate + (self.game_time / self.max_game_time) * 0.4
        # Bonus for energy efficiency and cooperation
        avg_hider_energy = sum(h.energy for h in self.hiders) / len(self.hiders)
        hider_fitness += (avg_hider_energy / 100) * 0.1

        seeker_fitness = (len(self.hiders) - survivors) / len(self.hiders)
        if self.game_time < self.max_game_time:
            seeker_fitness += (self.max_game_time - self.game_time) / self.max_game_time * 0.3
        
        # Bonus for seeker cooperation and efficiency
        avg_seeker_energy = sum(s.energy for s in self.seekers) / len(self.seekers)
        seeker_fitness += (avg_seeker_energy / 100) * 0.1

        # Store statistics
        self.hider_fitness_history.append(hider_fitness)
        self.seeker_fitness_history.append(seeker_fitness)
        self.survival_rates.append(survival_rate)

        # Evolve populations
        self.evolve_populations(hider_fitness, seeker_fitness)

        self.generation += 1
        self.start_new_game()

    def evolve_populations(self, hider_fitness, seeker_fitness):
        """Enhanced evolution with tournament selection"""
        # Tournament selection and crossover
        if hider_fitness > 0.6:
            self.hider_population = [self.mutate(genes, 0.03) for genes in self.hider_population]
        else:
            # More aggressive evolution when struggling
            new_hider_pop = []
            for _ in range(len(self.hider_population)):
                parent1 = random.choice(self.hider_population)
                parent2 = random.choice(self.hider_population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, 0.15)
                new_hider_pop.append(child)
            self.hider_population = new_hider_pop

        if seeker_fitness > 0.6:
            self.seeker_population = [self.mutate(genes, 0.03) for genes in self.seeker_population]
        else:
            new_seeker_pop = []
            for _ in range(len(self.seeker_population)):
                parent1 = random.choice(self.seeker_population)
                parent2 = random.choice(self.seeker_population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, 0.15)
                new_seeker_pop.append(child)
            self.seeker_population = new_seeker_pop

    def crossover(self, parent1, parent2):
        """Crossover two parent gene sets"""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
        else:
            child = parent1.copy()
        return child

    def mutate(self, genes, mutation_strength):
        """Enhanced mutation"""
        mutated = genes.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                change = random.gauss(0, mutation_strength)
                mutated[i] = max(0.1, mutated[i] + change)

                # Clamp values
                if i == 0:  # Speed
                    mutated[i] = max(0.5, min(6.0, mutated[i]))
                elif i == 1:  # Vision
                    mutated[i] = max(30, min(180, mutated[i]))
                else:  # 0-1 parameters
                    mutated[i] = max(0, min(1, mutated[i]))

        return mutated

    def draw_agent_vision(self, agent, genes, color):
        """Draw agent's vision range"""
        if self.show_vision:
            vision_surface = pygame.Surface((self.world_width, self.world_height))
            vision_surface.set_alpha(25)
            pygame.draw.circle(vision_surface, color,
                               (int(agent.x), int(agent.y)), int(genes[1]))
            self.screen.blit(vision_surface, (0, 0))

    def draw_trail(self, agent, color):
        """Draw agent's movement trail with fading effect"""
        if self.show_trails and len(agent.trail) > 1:
            for i in range(1, len(agent.trail)):
                alpha = int(150 * (i / len(agent.trail)))
                fade_color = (*color[:3], alpha)
                
                # Create a temporary surface for alpha blending
                trail_surface = pygame.Surface((5, 5))
                trail_surface.set_alpha(alpha)
                trail_surface.fill(color)
                
                pos = agent.trail[i]
                self.screen.blit(trail_surface, (pos[0]-2, pos[1]-2))

    def draw_world(self):
        """Draw the enhanced game world"""
        # Background
        world_rect = pygame.Rect(0, 0, self.world_width, self.world_height)
        pygame.draw.rect(self.screen, WHITE, world_rect)
        pygame.draw.rect(self.screen, BLACK, world_rect, 3)

        # Draw obstacles with enhanced visuals
        for obstacle in self.obstacles:
            color = BROWN if obstacle.movable else GRAY
            border_color = ORANGE if obstacle.being_pushed else DARK_GRAY
            
            # Draw obstacle
            pygame.draw.circle(self.screen, color, (int(obstacle.x), int(obstacle.y)), int(obstacle.radius))
            pygame.draw.circle(self.screen, border_color, (int(obstacle.x), int(obstacle.y)), int(obstacle.radius), 3)
            
            # Draw movable indicator
            if obstacle.movable:
                pygame.draw.circle(self.screen, YELLOW, (int(obstacle.x), int(obstacle.y)), 5)
            
            # Draw push direction
            if obstacle.being_pushed:
                end_x = obstacle.x + obstacle.push_direction[0] * 30
                end_y = obstacle.y + obstacle.push_direction[1] * 30
                pygame.draw.line(self.screen, RED, (obstacle.x, obstacle.y), (end_x, end_y), 4)

        # Draw vision ranges first (behind agents)
        for i, hider in enumerate(self.hiders):
            if hider.alive:
                self.draw_agent_vision(hider, self.hider_population[i], GREEN)

        for i, seeker in enumerate(self.seekers):
            self.draw_agent_vision(seeker, self.seeker_population[i], BLUE)

        # Draw trails
        for hider in self.hiders:
            if hider.alive:
                self.draw_trail(hider, GREEN)

        for seeker in self.seekers:
            self.draw_trail(seeker, BLUE)

        # Draw agents with enhanced visibility
        for i, hider in enumerate(self.hiders):
            if hider.alive:
                # Main body
                pygame.draw.circle(self.screen, GREEN, (int(hider.x), int(hider.y)), 12)
                pygame.draw.circle(self.screen, DARK_GREEN, (int(hider.x), int(hider.y)), 12, 3)
                
                # Energy bar
                energy_width = int(20 * (hider.energy / 100))
                pygame.draw.rect(self.screen, RED, (int(hider.x) - 10, int(hider.y) - 20, 20, 4))
                pygame.draw.rect(self.screen, GREEN, (int(hider.x) - 10, int(hider.y) - 20, energy_width, 4))
                
                # State indicator
                if self.show_states:
                    state_colors = {
                        "exploring": WHITE,
                        "hiding": RED,
                        "cooperating": PURPLE
                    }
                    state_color = state_colors.get(hider.strategic_state, WHITE)
                    pygame.draw.circle(self.screen, state_color, (int(hider.x), int(hider.y)), 4)
                
                # Direction indicator (small arrow)
                if len(hider.trail) > 1:
                    last_pos = hider.trail[-1]
                    second_last = hider.trail[-2]
                    dx = last_pos[0] - second_last[0]
                    dy = last_pos[1] - second_last[1]
                    if dx != 0 or dy != 0:
                        magnitude = math.sqrt(dx*dx + dy*dy)
                        if magnitude > 0:
                            dx /= magnitude
                            dy /= magnitude
                            arrow_end = (hider.x + dx * 15, hider.y + dy * 15)
                            pygame.draw.line(self.screen, BLACK, (hider.x, hider.y), arrow_end, 2)
            else:
                # Dead hider (X mark)
                pygame.draw.circle(self.screen, RED, (int(hider.x), int(hider.y)), 12)
                pygame.draw.line(self.screen, BLACK, 
                               (int(hider.x) - 8, int(hider.y) - 8), 
                               (int(hider.x) + 8, int(hider.y) + 8), 3)
                pygame.draw.line(self.screen, BLACK, 
                               (int(hider.x) + 8, int(hider.y) - 8), 
                               (int(hider.x) - 8, int(hider.y) + 8), 3)

        for i, seeker in enumerate(self.seekers):
            # Enhanced seeker visualization (larger triangle)
            points = [
                (seeker.x, seeker.y - 14),
                (seeker.x - 12, seeker.y + 10),
                (seeker.x + 12, seeker.y + 10)
            ]
            pygame.draw.polygon(self.screen, BLUE, points)
            pygame.draw.polygon(self.screen, BLACK, points, 3)
            
            # Energy bar
            energy_width = int(24 * (seeker.energy / 100))
            pygame.draw.rect(self.screen, RED, (int(seeker.x) - 12, int(seeker.y) - 25, 24, 4))
            pygame.draw.rect(self.screen, BLUE, (int(seeker.x) - 12, int(seeker.y) - 25, energy_width, 4))
            
            # State indicator
            if self.show_states:
                state_colors = {
                    "exploring": WHITE,
                    "hunting": RED,
                    "cooperating": PURPLE
                }
                state_color = state_colors.get(seeker.strategic_state, WHITE)
                pygame.draw.circle(self.screen, state_color, (int(seeker.x), int(seeker.y)), 4)
            
            # Vision cone (more visible)
            if self.show_vision and len(seeker.trail) > 1:
                last_pos = seeker.trail[-1]
                second_last = seeker.trail[-2]
                dx = last_pos[0] - second_last[0]
                dy = last_pos[1] - second_last[1]
                if dx != 0 or dy != 0:
                    magnitude = math.sqrt(dx*dx + dy*dy)
                    if magnitude > 0:
                        dx /= magnitude
                        dy /= magnitude
                        # Draw vision cone
                        cone_length = self.seeker_population[i][1] * 0.7  # Vision range
                        cone_width = 30
                        
                        cone_points = [
                            (seeker.x, seeker.y),
                            (seeker.x + dx * cone_length - dy * cone_width, seeker.y + dy * cone_length + dx * cone_width),
                            (seeker.x + dx * cone_length + dy * cone_width, seeker.y + dy * cone_length - dx * cone_width)
                        ]
                        
                        cone_surface = pygame.Surface((self.world_width, self.world_height))
                        cone_surface.set_alpha(40)
                        pygame.draw.polygon(cone_surface, CYAN, cone_points)
                        self.screen.blit(cone_surface, (0, 0))

    def draw_stats_panel(self):
        """Draw enhanced statistics panel"""
        panel_rect = pygame.Rect(self.world_width, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, LIGHT_GRAY, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)

        x_offset = self.world_width + 10
        y_offset = 20

        # Title
        title = self.font.render("ENHANCED AI EVOLUTION", True, BLACK)
        self.screen.blit(title, (x_offset, y_offset))
        y_offset += 40

        # Generation info
        gen_text = self.small_font.render(f"Generation: {self.generation}", True, BLACK)
        self.screen.blit(gen_text, (x_offset, y_offset))
        y_offset += 20

        time_text = self.small_font.render(f"Time: {self.game_time}/{self.max_game_time}", True, BLACK)
        self.screen.blit(time_text, (x_offset, y_offset))
        y_offset += 20

        survivors = sum(1 for h in self.hiders if h.alive)
        surv_text = self.small_font.render(f"Survivors: {survivors}/{len(self.hiders)}", True, BLACK)
        self.screen.blit(surv_text, (x_offset, y_offset))
        y_offset += 30

        # Agent states
        if self.show_states:
            states_title = self.small_font.render("Agent States:", True, BLACK)
            self.screen.blit(states_title, (x_offset, y_offset))
            y_offset += 20
            
            hider_states = {}
            for hider in self.hiders:
                if hider.alive:
                    state = hider.strategic_state
                    hider_states[state] = hider_states.get(state, 0) + 1
            
            for state, count in hider_states.items():
                state_text = self.small_font.render(f"H-{state}: {count}", True, GREEN)
                self.screen.blit(state_text, (x_offset, y_offset))
                y_offset += 15
            
            seeker_states = {}
            for seeker in self.seekers:
                state = seeker.strategic_state
                seeker_states[state] = seeker_states.get(state, 0) + 1
            
            for state, count in seeker_states.items():
                state_text = self.small_font.render(f"S-{state}: {count}", True, BLUE)
                self.screen.blit(state_text, (x_offset, y_offset))
                y_offset += 15
            
            y_offset += 20

        # Fitness history graph
        if len(self.hider_fitness_history) > 1:
            graph_rect = pygame.Rect(x_offset, y_offset, PANEL_WIDTH - 30, 120)
            pygame.draw.rect(self.screen, WHITE, graph_rect)
            pygame.draw.rect(self.screen, BLACK, graph_rect, 1)

            # Draw fitness lines
            max_gen = len(self.hider_fitness_history)
            for i in range(1, max_gen):
                # Hider fitness (green)
                x1 = x_offset + (i - 1) * (graph_rect.width / max(max_gen - 1, 1))
                y1 = y_offset + graph_rect.height - (self.hider_fitness_history[i - 1] * graph_rect.height)
                x2 = x_offset + i * (graph_rect.width / max(max_gen - 1, 1))
                y2 = y_offset + graph_rect.height - (self.hider_fitness_history[i] * graph_rect.height)
                pygame.draw.line(self.screen, GREEN, (x1, y1), (x2, y2), 2)

                # Seeker fitness (blue)
                y1_s = y_offset + graph_rect.height - (self.seeker_fitness_history[i - 1] * graph_rect.height)
                y2_s = y_offset + graph_rect.height - (self.seeker_fitness_history[i] * graph_rect.height)
                pygame.draw.line(self.screen, BLUE, (x1, y1_s), (x2, y2_s), 2)

            y_offset += 140

        # Enhanced gene display
        if self.show_genes and len(self.hider_population) > 0:
            genes_title = self.small_font.render("Best Genes:", True, BLACK)
            self.screen.blit(genes_title, (x_offset, y_offset))
            y_offset += 20

            # Best hider genes
            best_hider_idx = max(range(len(self.hiders)), 
                               key=lambda i: self.hiders[i].energy if self.hiders[i].alive else 0)
            best_hider_genes = self.hider_population[best_hider_idx]

            gene_names = ["Speed", "Vision", "Aggression", "Wall", "Memory", 
                         "Blocks", "Coop", "Strategy", "Energy", "Predict"]
            
            hider_title = self.small_font.render("Hider:", True, GREEN)
            self.screen.blit(hider_title, (x_offset, y_offset))
            y_offset += 15
            
            for i, (name, value) in enumerate(zip(gene_names, best_hider_genes)):
                if i < 5:  # First 5 genes
                    gene_text = self.small_font.render(f"{name}: {value:.2f}", True, BLACK)
                    self.screen.blit(gene_text, (x_offset, y_offset))
                    y_offset += 14
            
            for i, (name, value) in enumerate(zip(gene_names[5:], best_hider_genes[5:]), 5):
                gene_text = self.small_font.render(f"{name}: {value:.2f}", True, BLACK)
                self.screen.blit(gene_text, (x_offset + 90, y_offset - (i-5)*14))

            y_offset += 20

            # Best seeker genes
            best_seeker_genes = max(self.seeker_population, key=sum)
            
            seeker_title = self.small_font.render("Seeker:", True, BLUE)
            self.screen.blit(seeker_title, (x_offset, y_offset))
            y_offset += 15
            
            for i, (name, value) in enumerate(zip(gene_names, best_seeker_genes)):
                if i < 5:
                    gene_text = self.small_font.render(f"{name}: {value:.2f}", True, BLACK)
                    self.screen.blit(gene_text, (x_offset, y_offset))
                    y_offset += 14
            
            for i, (name, value) in enumerate(zip(gene_names[5:], best_seeker_genes[5:]), 5):
                gene_text = self.small_font.render(f"{name}: {value:.2f}", True, BLACK)
                self.screen.blit(gene_text, (x_offset + 90, y_offset - (i-5)*14))

        # Controls
        y_offset = WINDOW_HEIGHT - 180
        controls_title = self.small_font.render("Controls:", True, BLACK)
        self.screen.blit(controls_title, (x_offset, y_offset))
        y_offset += 20

        controls = [
            "SPACE: Pause/Resume",
            "V: Toggle Vision",
            "T: Toggle Trails", 
            "G: Toggle Genes",
            "S: Toggle States",
            "R: Reset Simulation",
            "+/-: Speed Control",
            "ESC: Quit"
        ]

        for control in controls:
            control_text = self.small_font.render(control, True, BLACK)
            self.screen.blit(control_text, (x_offset, y_offset))
            y_offset += 15

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_vision = not self.show_vision
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_g:
                    self.show_genes = not self.show_genes
                elif event.key == pygame.K_s:
                    self.show_states = not self.show_states
                elif event.key == pygame.K_r:
                    self.__init__()  # Reset simulation
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed_multiplier = min(3.0, self.speed_multiplier + 0.2)
                elif event.key == pygame.K_MINUS:
                    self.speed_multiplier = max(0.2, self.speed_multiplier - 0.2)

        return True

    def run(self):
        """Main game loop"""
        running = True

        while running:
            running = self.handle_events()

            if not self.paused:
                self.update_game()

            # Draw everything
            self.screen.fill(BLACK)
            self.draw_world()
            self.draw_stats_panel()

            # Show pause indicator
            if self.paused:
                pause_text = self.font.render("PAUSED", True, YELLOW)
                pause_rect = pause_text.get_rect(center=(self.world_width // 2, 30))
                pygame.draw.rect(self.screen, BLACK, pause_rect.inflate(20, 10))
                self.screen.blit(pause_text, pause_rect)

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()


def main():
    """Run the enhanced pygame hide and seek simulation"""
    print("🚀 Enhanced AI Hide and Seek Evolution!")
    print("=" * 60)
    print("NEW FEATURES:")
    print("• Intelligent block manipulation")
    print("• Advanced AI strategies (cooperation, prediction)")
    print("• Enhanced agent visibility and states")
    print("• Energy system and strategic thinking")
    print("• Improved evolution with crossover")
    print("=" * 60)
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  V: Toggle vision ranges")
    print("  T: Toggle movement trails")
    print("  G: Toggle gene display")
    print("  S: Toggle agent states")
    print("  R: Reset simulation")
    print("  +/-: Adjust speed")
    print("  ESC: Quit")
    print("=" * 60)

    try:
        game = PyGameHideSeek()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure pygame is installed: pip install pygame")


if __name__ == "__main__":
    main()
