import numpy as np
import matplotlib.pyplot as plt

def draw_fractal(ax, start, angle, length, depth, color='gray'):
    """Draw a single branch of the fractal tree"""
    if depth == 0:
        return []
    
    # Calculate end point
    end_x = start[0] + length * np.cos(np.radians(angle))
    end_y = start[1] + length * np.sin(np.radians(angle))
    end = (end_x, end_y)
    
    # Draw the branch
    linewidth = depth * 0.8
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)
    
    # Store line segment with thickness info
    segments = [(start, end, linewidth)]
    
    # Recursively draw branches
    new_length = length * 0.75
    angle_change = 30
    
    left_segments = draw_fractal(ax, end, angle + angle_change, new_length, depth - 1, color)
    right_segments = draw_fractal(ax, end, angle - angle_change, new_length, depth - 1, color)
    
    return segments + left_segments + right_segments

def generate_scatter_points(center, radius, n_points):
    """Generate scattered points around center with upward bias"""
    angles = np.random.uniform(0, 2*np.pi, n_points)
    distances = np.random.normal(radius/2, radius/4, n_points)
    x = center[0] + distances * np.cos(angles)
    y = center[1] + distances * np.sin(angles) + radius/2  # Added upward bias
    return x, y

def point_to_cell(point, x_min, y_min, cell_size):
    """Convert point coordinates to grid cell indices"""
    i = int((point[1] - y_min) / cell_size)
    j = int((point[0] - x_min) / cell_size)
    return i, j

def mark_cells_within_radius(grid, x, y, radius, cell_size, bounds):
    """Mark all cells within given radius of a point"""
    radius_in_cells = int(np.ceil(radius / cell_size))
    x_min, _, y_min, _ = bounds
    
    center_i = int((y - y_min) / cell_size)
    center_j = int((x - x_min) / cell_size)
    
    for di in range(-radius_in_cells, radius_in_cells + 1):
        for dj in range(-radius_in_cells, radius_in_cells + 1):
            i, j = center_i + di, center_j + dj
            if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                # Check if cell center is within radius of point
                cell_center_x = x_min + (j + 0.5) * cell_size
                cell_center_y = y_min + (i + 0.5) * cell_size
                if np.hypot(cell_center_x - x, cell_center_y - y) <= radius:
                    grid[i, j] = 1

def mark_cells_along_line(grid, start, end, thickness, cell_size, bounds):
    """Mark all cells that a line segment passes through using Bresenham's algorithm plus thickness"""
    x_min, _, y_min, _ = bounds
    
    # Get start and end cell coordinates
    start_i = int((start[1] - y_min) / cell_size)
    start_j = int((start[0] - x_min) / cell_size)
    end_i = int((end[1] - y_min) / cell_size)
    end_j = int((end[0] - x_min) / cell_size)
    
    # Calculate line parameters
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.hypot(dx, dy)
    
    if length == 0:
        return
    
    # Calculate perpendicular unit vector for thickness
    if length > 0:
        ux = dx / length  # Unit vector along line
        uy = dy / length
        vx = -uy  # Perpendicular unit vector
        vy = ux
    else:
        vx, vy = 1, 0
    
    # Number of cells for line thickness (round up to ensure coverage)
    cells_thick = max(1, int(np.ceil(thickness / cell_size)))
    
    # Sample points along the line with proper thickness
    steps = max(abs(end_i - start_i), abs(end_j - start_j)) * 4 + 1  # Increased sampling
    for t in np.linspace(0, 1, steps):
        # Center point along line
        cx = start[0] + t * dx
        cy = start[1] + t * dy
        
        # Mark cells within thickness perpendicular to line
        for k in np.linspace(-thickness/2, thickness/2, cells_thick*2 + 1):
            px = cx + k * vx
            py = cy + k * vy
            
            i, j = int((py - y_min) / cell_size), int((px - x_min) / cell_size)
            if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                grid[i, j] = 1
                
                # Also mark immediate neighbors to ensure connectivity
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        if np.hypot((px - x_min - (nj + 0.5)*cell_size),
                                  (py - y_min - (ni + 0.5)*cell_size)) <= thickness/2:
                            grid[ni, nj] = 1

def main():
    # Parameters
    resolution = 100  # Increased grid resolution
    start_pos = (0, -60)  # Moved starting position even further down
    start_length = 40
    max_depth = 7
    n_scatter = 1000
    
    # Calculate bounds to ensure square aspect ratio and prevent cutoff
    margin = start_length * 2.5  # Increased margin to prevent cutoff
    bounds = [-margin, margin, -margin, margin]
    cell_size = (bounds[1] - bounds[0]) / resolution
    
    # Create figure with square aspect ratio
    plt.figure(figsize=(15, 5))
    
    # 1. Draw fractal with scatter
    ax1 = plt.subplot(131, aspect='equal')
    ax1.set_title("Fractal Tree with Scatter")
    
    # Draw fractal and collect line segments with thickness
    fractal_segments = draw_fractal(ax1, start_pos, 90, start_length, max_depth, 'gray')
    
    # Generate and plot scatter points
    scatter_x, scatter_y = generate_scatter_points(start_pos, start_length*3, n_scatter)
    ax1.scatter(scatter_x, scatter_y, color='red', alpha=0.4, s=10, zorder=10)
    
    # 2. Ground Truth Grid
    ax2 = plt.subplot(132, aspect='equal')
    ax2.set_title("Ground Truth Grid")
    
    # Create grid
    grid = np.zeros((resolution, resolution))
    
    # Mark ground truth cells considering branch width
    for start, end, thickness in fractal_segments:
        mark_cells_along_line(grid, start, end, thickness/20, cell_size, bounds)  # Reduced thickness by factor of 20
    
    # Plot ground truth grid
    x_edges = np.linspace(bounds[0], bounds[1], resolution + 1)
    y_edges = np.linspace(bounds[2], bounds[3], resolution + 1)
    
    ax2.pcolormesh(x_edges, y_edges, grid, 
                   cmap=plt.matplotlib.colors.ListedColormap(['white', '#0000FF']), 
                   alpha=0.3)
    
    # Add thin grid lines
    for x in x_edges:
        ax2.axvline(x=x, color='gray', alpha=0.2, linewidth=0.1)
    for y in y_edges:
        ax2.axhline(y=y, color='gray', alpha=0.2, linewidth=0.1)
    
    # 3. Hit/Miss Visualization
    ax3 = plt.subplot(133, aspect='equal')
    ax3.set_title("Hit/Miss Analysis")
    
    # Create visualization grid starting with ground truth
    vis_grid = grid.copy()  # Start with ground truth (will be 1s and 0s)
    hit_cells = set()
    miss_cells = set()
    
    # Mark hits and misses
    for x, y in zip(scatter_x, scatter_y):
        i, j = point_to_cell((x, y), bounds[0], bounds[2], cell_size)
        if 0 <= i < resolution and 0 <= j < resolution:
            if grid[i, j] > 0:
                vis_grid[i, j] = 2  # Hit
                hit_cells.add((i, j))
            else:
                vis_grid[i, j] = 3  # Miss
                miss_cells.add((i, j))
    
    # Plot hit/miss grid with ground truth visible
    colors = ['white', '#0000FF', '#00FF00', '#FF0000']  # White, Ground Truth, Hit, Miss
    ax3.pcolormesh(x_edges, y_edges, vis_grid, 
                   cmap=plt.matplotlib.colors.ListedColormap(colors), 
                   alpha=0.3)
    
    # Add thin grid lines
    for x in x_edges:
        ax3.axvline(x=x, color='gray', alpha=0.2, linewidth=0.1)
    for y in y_edges:
        ax3.axhline(y=y, color='gray', alpha=0.2, linewidth=0.1)
    
    # Set bounds and remove axis numbers for all plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    plt.tight_layout()
    
    # Calculate metrics
    ground_truth_cells = np.sum(grid > 0)
    n_hits = len(hit_cells)
    n_misses = len(miss_cells)
    mandala_score = n_hits / ground_truth_cells if ground_truth_cells > 0 else 0
    
    # Add stats only to the last plot
    stats_text = f"Ground Truth: {ground_truth_cells}\nHits: {n_hits}\nMisses: {n_misses}\nMandala Score: {mandala_score:.3f}"
    ax3.text(0.02, 0.98, stats_text,
             transform=ax3.transAxes,
             verticalalignment='top',
             fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.savefig('fractal_analysis_step_by_step_new.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 