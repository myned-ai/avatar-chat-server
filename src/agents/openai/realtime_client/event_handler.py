"""
Event handler base class for Realtime API.

Provides event registration, dispatch, and handling functionality
inherited by RealtimeAPI and RealtimeClient.
"""

import asyncio
from typing import Callable, Dict, List, Any, Optional
from collections import defaultdict


# Type alias for event handler callbacks
EventHandlerCallback = Callable[[Any], Any]


class RealtimeEventHandler:
    """
    Base class for event handling in Realtime API.
    Provides on(), off(), onNext(), waitForNext(), and dispatch() methods.
    """
    
    def __init__(self):
        self._event_handlers: Dict[str, List[EventHandlerCallback]] = defaultdict(list)
        self._next_event_handlers: Dict[str, List[EventHandlerCallback]] = defaultdict(list)
        self._event_waiters: Dict[str, List[asyncio.Future]] = defaultdict(list)
    
    def clear_event_handlers(self) -> bool:
        """
        Clears all event handlers.
        
        Returns:
            True on success
        """
        self._event_handlers.clear()
        self._next_event_handlers.clear()
        return True
    
    def on(self, event_name: str, callback: EventHandlerCallback) -> EventHandlerCallback:
        """
        Listen to specific events.
        
        Args:
            event_name: The name of the event to listen to
            callback: Function to execute when event fires
            
        Returns:
            The callback function (for chaining/removal)
        """
        self._event_handlers[event_name].append(callback)
        return callback
    
    def off(self, event_name: str, callback: Optional[EventHandlerCallback] = None) -> bool:
        """
        Stop listening to a specific event.
        
        Args:
            event_name: The name of the event to stop listening to
            callback: Specific callback to remove (if None, removes all)
            
        Returns:
            True on success
        """
        if callback is None:
            self._event_handlers[event_name] = []
        elif callback in self._event_handlers[event_name]:
            self._event_handlers[event_name].remove(callback)
        return True
    
    def on_next(self, event_name: str, callback: EventHandlerCallback) -> EventHandlerCallback:
        """
        Listen for the next event of a specified type (one-time handler).
        
        Args:
            event_name: The name of the event to listen to
            callback: Function to execute on event
            
        Returns:
            The callback function
        """
        self._next_event_handlers[event_name].append(callback)
        return callback
    
    def off_next(self, event_name: str, callback: Optional[EventHandlerCallback] = None) -> bool:
        """
        Stop listening for the next event of a specified type.
        
        Args:
            event_name: The name of the event
            callback: Specific callback to remove (if None, removes all)
            
        Returns:
            True on success
        """
        if callback is None:
            self._next_event_handlers[event_name] = []
        elif callback in self._next_event_handlers[event_name]:
            self._next_event_handlers[event_name].remove(callback)
        return True
    
    async def wait_for_next(self, event_name: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for the next occurrence of a specific event.
        
        Args:
            event_name: The name of the event to wait for
            timeout: Maximum time to wait in seconds (None = no timeout)
            
        Returns:
            The event data when the event fires
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._event_waiters[event_name].append(future)
        
        try:
            if timeout:
                return await asyncio.wait_for(future, timeout)
            else:
                return await future
        except asyncio.TimeoutError:
            # Remove the future from waiters if timeout
            if future in self._event_waiters[event_name]:
                self._event_waiters[event_name].remove(future)
            raise
    
    def dispatch(self, event_name: str, event: Any = None) -> bool:
        """
        Dispatch an event to all registered handlers.
        
        Args:
            event_name: The name of the event to dispatch
            event: The event data to pass to handlers
            
        Returns:
            True on success
        """
        # Handle regular event handlers
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                result = handler(event)
                # If the handler is a coroutine, schedule it
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                print(f"Error in event handler for '{event_name}': {e}")
        
        # Handle one-time next event handlers
        next_handlers = self._next_event_handlers.pop(event_name, [])
        for handler in next_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                print(f"Error in next event handler for '{event_name}': {e}")
        
        # Handle waiters
        waiters = self._event_waiters.pop(event_name, [])
        for future in waiters:
            if not future.done():
                future.set_result(event)
        
        return True
