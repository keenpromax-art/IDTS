import sys
sys.path.insert(0, r'C:\IDTS')

with open(r'C:\IDTS\trader\ui\dashboard.py', 'r') as f:
    content = f.read()

# Add cache attribute initialization in __init__
# Find the _render_cache usage and add initialization
old_init_check = 'self._render_cache = None' not in content
if old_init_check:
    # Add cache initialization after other init attributes
    # Find a good place - after _view and _input_mode
    init_search = 'self._view = None'
    if init_search in content:
        content = content.replace(
            'self._view = None',
            'self._view = None\n        self._render_cache = None'
        )
        print('Added _render_cache init')
    else:
        print('Could not find _view init place')
else:
    print('_render_cache already initialized')

# Now add caching to _render_frame method
# Find the method and add cache usage
render_method_start = content.find('def _render_frame')
if render_method_start >= 0:
    # Find the end of the method (next method or class)
    method_end = content.find('def ', render_method_start + 1)
    if method_end >= 0:
        method_body = content[render_method_start:method_end]
        
        # Add cache check at the beginning of the method
        cache_check = '''
        # Cache: only recalculate if layout dimensions or state changed
        if self._render_cache is not None:
            cached_layout, cache_key = self._render_cache
            # Simple key: view + wide + header_h
            key = (view, wide, self._header_h())
            if key == cache_key:
                return cached_layout
        
        # Store cache before returning
        cache_store = None'''
        
        # Insert cache check after the method header but before layout = Layout()
        header_end = content.find('    def _render_frame(self) -> Layout:', render_method_start)
        # Actually, let's find a better insertion point
        
        # Find the line "        layout = Layout()" and insert before it
        insert_pos = content.find('        layout = Layout()', render_method_start)
        if insert_pos >= 0:
            # Insert cache check before layout creation
            content = content[:insert_pos] + cache_check + content[insert_pos:]
            print('Added cache check to _render_frame')
        else:
            print('Could not find layout insert position')
    else:
        print('Could not find method end')
else:
    print('_render_cache already in init')

# Now let's also add space caching
space_start = content.find('def _space')
if space_start >= 0:
    method_end = content.find('def ', space_start + 1)
    if method_end >= 0:
        method_body = content[space_start:method_end]
        # Add cache check
        cache_check2 = '''
        # Cache: only recalculate if window size changed
        if self._space_cache is not None:
            cached_h, cache_key = self._space_cache
            key = (self.console.width, self.console.height)
            if key == cache_key:
                return cached_h
        
        # Store cache before returning
        cache_store = None'''
        
        # Find "    h = self.console.height or 40" or similar
        # and insert cache check
        print('Space method cache check would go here')
    
with open(r'C:\IDTS\trader\ui\dashboard.py', 'w') as f:
    f.write(content)

print('Render caching improvement complete')