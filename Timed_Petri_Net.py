import heapq
from typing import Dict, List, Callable

# ----------------------------
# Petri Net Components
# ----------------------------

class Place:
    def __init__(self, name: str, tokens: int = 0):
        self.name = name
        self.tokens = tokens

    def add_tokens(self, count: int):
        self.tokens += count

    def remove_tokens(self, count: int):
        if self.tokens < count:
            raise ValueError(f"Not enough tokens in {self.name} to remove {count}.")
        self.tokens -= count

    def __repr__(self):
        return f"Place(name='{self.name}', tokens={self.tokens})"


class CombinedTransition:
    """
    A transition that fires with a fixed delay and can fire in bursts.
    
    For burst mode:
      - burst_size > 1 means that when enabled the transition will fire burst_size times
        with a fixed delay between firings, without rechecking the enabling condition in between.
      - The enabling condition requires that each input place have tokens >= (arc weight * burst_size).
      
    Attributes:
      - fixed_delay: Time interval between consecutive firings (or burst events) in deterministic mode.
      - burst_size: Number of firings per burst.
      - deterministic: When True, always use fixed_delay; no random component.
      - force_schedule: When True, the transition will always be scheduled regardless of enabling condition.
      - poll_only: When True, the transition will be scheduled but will only update tokens if enabled.
    """
    def __init__(self, name: str,
                 input_places: Dict[Place, int],
                 output_places: Dict[Place, int],
                 fixed_delay: float,
                 burst_size: int = 1,
                 deterministic: bool = False,
                 force_schedule: bool = False,
                 poll_only: bool = False):
        self.name = name
        self.input_places = input_places    # e.g., {place: tokens_required per firing}
        self.output_places = output_places  # e.g., {place: tokens_to_add per firing}
        self.fixed_delay = fixed_delay
        self.burst_size = burst_size
        self.deterministic = deterministic
        self.force_schedule = force_schedule  # Always schedule even if not enabled
        self.poll_only = poll_only  # Only check condition when firing, not when scheduling
        # For burst mode: track remaining events in the current burst.
        self.burst_remaining = 0

    def enabling_threshold(self) -> Dict[Place, int]:
        """Return the required tokens for each input to start a burst."""
        factor = self.burst_size if self.burst_size > 1 else 1
        return {place: weight * factor for place, weight in self.input_places.items()}

    def is_enabled(self) -> bool:
        """Check enabling condition.
        
        For burst-mode (if not currently in a burst), check that each input has tokens 
        >= (arc weight * burst_size). Otherwise, for single firing, check tokens >= arc weight.
        
        If force_schedule is True, this always returns True for scheduling purposes.
        """
        if self.force_schedule:
            return True
            
        if self.burst_size > 1 and self.burst_remaining == 0:
            threshold = self.enabling_threshold()
            for place, req in threshold.items():
                if place.tokens < req:
                    print(f"Transition '{self.name}' not enabled (burst check): {place.name} has {place.tokens} tokens, requires {req}.")
                    return False
            return True
        else:
            for place, weight in self.input_places.items():
                if place.tokens < weight:
                    print(f"Transition '{self.name}' not enabled (single check): {place.name} has {place.tokens} tokens, requires {weight}.")
                    return False
            return True

    def can_actually_fire(self) -> bool:
        """Check if the transition can actually fire based on token counts.
        Used for poll_only transitions to determine if tokens should be modified."""
        for place, weight in self.input_places.items():
            if place.tokens < weight:
                return False
        return True
    
    def start_burst(self):
        """Start a new burst."""
        self.burst_remaining = self.burst_size

    def fire(self):
        """Fire one firing event."""
        # For poll_only transitions, first check if we actually can fire
        if self.poll_only:
            if not self.can_actually_fire():
                print(f"Transition '{self.name}' polled but condition not met.")
                return False
                
        # Safety check for a single firing.
        for place, weight in self.input_places.items():
            if place.tokens < weight:
                raise Exception(f"Transition '{self.name}' cannot fire due to insufficient tokens in {place.name}.")
                
        for place, weight in self.input_places.items():
            place.remove_tokens(weight)
        for place, count in self.output_places.items():
            place.add_tokens(count)
        print(f"Transition '{self.name}' fired.")
        return True

    def scheduled_time(self, current_time: float) -> float:
        """In deterministic mode, next time is current_time + fixed_delay."""
        return current_time + self.fixed_delay


# ----------------------------
# Persistent Event Scheduling Petri Net Simulator
# ----------------------------

class PetriNet:
    """
    The simulation engine maintains an event queue for transitions.
    
    For burst-mode transitions, once a burst is started, the firing events occur at fixed intervals
    without rechecking the enabling condition until the burst completes.
    """
    def __init__(self, places: List[Place], transitions: List[CombinedTransition]):
        self.places = places
        self.transitions = transitions
        self.current_time = 0.0
        self.event_queue = []
        self.event_counter = 0   # Tie-breaker counter for events.

    def schedule_transition(self, transition: CombinedTransition, scheduled_time: float, burst_event: bool = False):
        self.event_counter += 1
        heapq.heappush(self.event_queue, (scheduled_time, self.event_counter, transition, burst_event))
        kind = "burst" if burst_event else "regular"
        print(f"Scheduled {kind} event for '{transition.name}' at time {scheduled_time:.3f}")
    
    def initialize_events(self):
        for t in self.transitions:
            if t.force_schedule or t.is_enabled():
                if t.burst_size > 1:
                    t.start_burst()
                    self.schedule_transition(t, t.scheduled_time(self.current_time), burst_event=True)
                else:
                    self.schedule_transition(t, t.scheduled_time(self.current_time))
            else:
                print(f"Transition '{t.name}' is not enabled at time {self.current_time:.3f}; not scheduled.")
    
    def simulate1(self, max_time: float):
        self.initialize_events()
        while self.event_queue and self.current_time <= max_time:
            scheduled_time, _, transition, burst_event = heapq.heappop(self.event_queue)
            self.current_time = scheduled_time
            print(f"\n--- At time {self.current_time:.3f} ---")
            
            if burst_event:
                # Burst event: fire without re-checking enabling condition.
                fired = transition.fire()
                if fired or transition.poll_only:
                    transition.burst_remaining -= 1
                self.print_places()
                
                if transition.burst_remaining > 0:
                    next_time = transition.scheduled_time(self.current_time)
                    self.schedule_transition(transition, next_time, burst_event=True)
                else:
                    if transition.force_schedule or transition.is_enabled():
                        transition.start_burst()
                        next_time = transition.scheduled_time(self.current_time)
                        self.schedule_transition(transition, next_time, burst_event=True)
                    else:
                        print(f"Transition '{transition.name}' is no longer enabled after completing a burst.")
            else:
                # Individual transition event.
                if transition.force_schedule or transition.is_enabled():
                    fired = transition.fire()
                    self.print_places()
                    
                    # Always reschedule forced transitions
                    if transition.force_schedule or transition.is_enabled():
                        next_time = transition.scheduled_time(self.current_time)
                        self.schedule_transition(transition, next_time)
                else:
                    print(f"Transition '{transition.name}' was scheduled at time {self.current_time:.3f} but is not enabled.")
        
        print("Simulation ended.")
    
    def print_places(self):
        for place in self.places:
            print(place)
    
    def simulate_until(self, stop_condition: Callable[['PetriNet'], bool], max_time: float = float('inf')):
        """
        Simulate the Petri net until a condition is met or max_time is reached.
        
        Args:
            stop_condition: A function that takes the PetriNet instance and returns True 
                           when the simulation should stop.
            max_time: Maximum simulation time (optional safety limit)
        
        Returns:
            The simulation time when the condition was met, or None if max_time was reached
            without meeting the condition.
        """
        self.initialize_events()
        
        # Check if the condition is already met at the start
        if stop_condition(self):
            print(f"Stop condition met at time {self.current_time:.3f} before simulation started.")
            return self.current_time
        
        while self.event_queue and self.current_time <= max_time:
            scheduled_time, _, transition, burst_event = heapq.heappop(self.event_queue)
            self.current_time = scheduled_time
            print(f"\n--- At time {self.current_time:.3f} ---")
            
            if burst_event:
                # Burst event: fire without re-checking enabling condition.
                fired = transition.fire()
                if fired or transition.poll_only:
                    transition.burst_remaining -= 1
                self.print_places()
                
                # Check if the condition is met after firing
                if stop_condition(self):
                    print(f"Stop condition met at time {self.current_time:.3f}")
                    return self.current_time
                
                if transition.burst_remaining > 0:
                    next_time = transition.scheduled_time(self.current_time)
                    self.schedule_transition(transition, next_time, burst_event=True)
                else:
                    if transition.force_schedule or transition.is_enabled():
                        transition.start_burst()
                        next_time = transition.scheduled_time(self.current_time)
                        self.schedule_transition(transition, next_time, burst_event=True)
                    else:
                        print(f"Transition '{transition.name}' is no longer enabled after completing a burst.")
            else:
                # Individual transition event.
                if transition.force_schedule or transition.is_enabled():
                    fired = transition.fire()
                    self.print_places()
                    
                    # Check if the condition is met after firing
                    if stop_condition(self):
                        print(f"Stop condition met at time {self.current_time:.3f}")
                        return self.current_time
                    
                    # Always reschedule forced transitions
                    if transition.force_schedule or transition.is_enabled():
                        next_time = transition.scheduled_time(self.current_time)
                        self.schedule_transition(transition, next_time)
                        
                    # Check if any transitions have become newly enabled
                    for t in self.transitions:
                        if t != transition and t not in [event[2] for event in self.event_queue] and t.is_enabled():
                            self.schedule_transition(t, t.scheduled_time(self.current_time))
                else:
                    print(f"Transition '{transition.name}' was scheduled at time {self.current_time:.3f} but is not enabled.")
        
        if self.current_time > max_time:
            print(f"Simulation reached max_time ({max_time:.3f}) without meeting the stop condition.")
        else:
            print("Simulation ended without meeting the stop condition (event queue empty).")
        
        return None
    def simulate(self, max_time: float):
        self.initialize_events()
        while self.event_queue and self.current_time <= max_time:
            scheduled_time, _, transition, burst_event = heapq.heappop(self.event_queue)
            self.current_time = scheduled_time
            print(f"\n--- At time {self.current_time:.3f} ---")
            
            if burst_event:
                # Burst event: fire without re-checking enabling condition.
                fired = transition.fire()
                if fired or transition.poll_only:
                    transition.burst_remaining -= 1
                self.print_places()
                
                if transition.burst_remaining > 0:
                    next_time = transition.scheduled_time(self.current_time)
                    self.schedule_transition(transition, next_time, burst_event=True)
                else:
                    if transition.force_schedule or transition.is_enabled():
                        transition.start_burst()
                        next_time = transition.scheduled_time(self.current_time)
                        self.schedule_transition(transition, next_time, burst_event=True)
                    else:
                        print(f"Transition '{transition.name}' is no longer enabled after completing a burst.")
            else:
                # Individual transition event.
                if transition.force_schedule or transition.is_enabled():
                    fired = transition.fire()
                    self.print_places()
                    
                    # Always reschedule forced transitions
                    if transition.force_schedule or transition.is_enabled():
                        next_time = transition.scheduled_time(self.current_time)
                        self.schedule_transition(transition, next_time)
                        
                    # Check if any transitions have become newly enabled
                    for t in self.transitions:
                        if t != transition and t not in [event[2] for event in self.event_queue] and t.is_enabled():
                            self.schedule_transition(t, t.scheduled_time(self.current_time))
                else:
                    print(f"Transition '{transition.name}' was scheduled at time {self.current_time:.3f} but is not enabled.")
    