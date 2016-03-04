package za.co.luma.math.sampling;

import java.util.LinkedList;
import java.util.List;

import za.co.iocom.math.MathUtil;
import za.co.luma.geom.Vector2DDouble;
import za.co.luma.geom.Vector2DInt;
import za.co.luma.math.function.RealFunction2DDouble;

/**
 * Algorithm based on <emph>Fast Poisson Disk Sampling in Arbitrary Dimensions</emph> by Robert Bridson, but with an
 * arbitrary minimum distance function. See also the paper <emph>A Spatial Data Structure for Fast Poisson-Disk Sample
 * Generation</emph> Daniel Dunbar and Greg Humphreys for other algorithms and a comparrisson.
 * 
 * @author Herman Tulleken
 */
public class PoissonDiskSampler implements Sampler<Vector2DDouble>
{
	private final static int DEFAULT_POINTS_TO_GENERATE = 30;
	private final int pointsToGenerate; // k in literature
	private final Vector2DDouble p0, p1;
	private final Vector2DDouble dimensions;
	private final double cellSize; // r / sqrt(n), for 2D: r / sqrt(2)
	private final double minDist; // r
	private final int gridWidth, gridHeight;

	/**
	 * A safety measure - no more than this number of points are produced by ther algorithm.
	 */
	public final static int MAX_POINTS = 100000;

	private RealFunction2DDouble distribution;

	/**
	 * Construct a new PoissonDisk object, with a given domain and minimum distance between points.
	 * 
	 * @param x0
	 *            x-coordinate of bottom left corner of domain.
	 * @param y0
	 *            x-coordinate of bottom left corner of domain.
	 * @param x1
	 *            x-coordinate of bottom left corner of domain.
	 * @param y1
	 *            x-coordinate of bottom left corner of domain.
	 * 
	 * @param distribution
	 *            A function that gives the minimum radius between points in the vicinity of a point.
	 */
	public PoissonDiskSampler(double x0, double y0, double x1, double y1, double minDist, RealFunction2DDouble distribution, int pointsToGenerate)
	{
		p0 = new Vector2DDouble(x0, y0);
		p1 = new Vector2DDouble(x1, y1);
		dimensions = new Vector2DDouble(x1 - x0, y1 - y0);

		this.minDist = minDist;
		this.distribution = distribution;
		this.pointsToGenerate = pointsToGenerate;
		cellSize = minDist / Math.sqrt(2);
		gridWidth = (int) (dimensions.x / cellSize) + 1;
		gridHeight = (int) (dimensions.y / cellSize) + 1;
	}
	/**
	 * Construct a new PoissonDisk object, with a given domain and minimum distance between points.
	 * 
	 * @param x0
	 *            x-coordinate of bottom left corner of domain.
	 * @param y0
	 *            x-coordinate of bottom left corner of domain.
	 * @param x1
	 *            x-coordinate of bottom left corner of domain.
	 * @param y1
	 *            x-coordinate of bottom left corner of domain.
	 * 
	 * @param distribution
	 *            A function that gives the minimum radius between points in the vicinity of a point.
	 */
	public PoissonDiskSampler(double x0, double y0, double x1, double y1, double minDist,
			RealFunction2DDouble distribution)
	{
		this(x0, y0, x1, y1, minDist, distribution, DEFAULT_POINTS_TO_GENERATE);
	}
	/**
	 * Generates a list of points following the Poisson distribution. No more than MAX_POINTS are produced.
	 * 
	 * @return The sample set.
	 */
	@SuppressWarnings("unchecked")
	public List<Vector2DDouble> sample()
	{
		List<Vector2DDouble> activeList = new LinkedList<Vector2DDouble>();
		List<Vector2DDouble> pointList = new LinkedList<Vector2DDouble>();
		List<Vector2DDouble> grid[][] = new List[gridWidth][gridHeight];

		for (int i = 0; i < gridWidth; i++)
		{
			for (int j = 0; j < gridHeight; j++)
			{
				grid[i][j] = new LinkedList<Vector2DDouble>();
			}
		}

		addFirstPoint(grid, activeList, pointList);

		while (!activeList.isEmpty() && (pointList.size() < MAX_POINTS))
		{
			int listIndex = MathUtil.random.nextInt(activeList.size());

			Vector2DDouble point = activeList.get(listIndex);
			boolean found = false;

			for (int k = 0; k < pointsToGenerate; k++)
			{
				found |= addNextPoint(grid, activeList, pointList, point);
			}

			if (!found)
			{
				activeList.remove(listIndex);
			}
		}

		return pointList;
	}

	private boolean addNextPoint(List<Vector2DDouble>[][] grid, List<Vector2DDouble> activeList,
			List<Vector2DDouble> pointList, Vector2DDouble point)
	{
		boolean found = false;
		double fraction = distribution.getDouble((int) point.x, (int) point.y);
		Vector2DDouble q = generateRandomAround(point, fraction * minDist);

		if ((q.x >= p0.x) && (q.x < p1.x) && (q.y > p0.y) && (q.y < p1.y))
		{
			Vector2DInt qIndex = pointDoubleToInt(q, p0, cellSize);

			boolean tooClose = false;

			for (int i = Math.max(0, qIndex.x - 2); (i < Math.min(gridWidth, qIndex.x + 3)) && !tooClose; i++)
			{
				for (int j = Math.max(0, qIndex.y - 2); (j < Math.min(gridHeight, qIndex.y + 3)) && !tooClose; j++)
				{
					for (Vector2DDouble gridPoint : grid[i][j])
					{
						if (Vector2DDouble.distance(gridPoint, q) < minDist * fraction)
						{
							tooClose = true;
						}
					}
				}
			}

			if (!tooClose)
			{
				found = true;
				activeList.add(q);
				pointList.add(q);
				grid[qIndex.x][qIndex.y].add(q);
			}
		}
		
		return found;
	}

	private void addFirstPoint(List<Vector2DDouble>[][] grid, List<Vector2DDouble> activeList,
			List<Vector2DDouble> pointList)
	{
		double d = MathUtil.random.nextDouble();
		double xr = p0.x + dimensions.x * (d);

		d = MathUtil.random.nextDouble();
		double yr = p0.y + dimensions.y * (d);

		Vector2DDouble p = new Vector2DDouble(xr, yr);
		Vector2DInt index = pointDoubleToInt(p, p0, cellSize);

		grid[index.x][index.y].add(p);
		activeList.add(p);
		pointList.add(p);
	}

	/**
	 * Converts a PointDouble to a PointInt that represents the index coordinates of the point in the background grid.
	 */
	static Vector2DInt pointDoubleToInt(Vector2DDouble pointDouble, Vector2DDouble origin, double cellSize)
	{
		return new Vector2DInt((int) ((pointDouble.x - origin.x) / cellSize),
				(int) ((pointDouble.y - origin.y) / cellSize));
	}

	/**
	 * Generates a random point in the analus around the given point. The analus has inner radius minimum distance and
	 * outer radius twice that.
	 * 
	 * @param centre
	 *            The point around which the random point should be.
	 * @return A new point, randomly selected.
	 */
	static Vector2DDouble generateRandomAround(Vector2DDouble centre, double minDist)
	{
		double d = MathUtil.random.nextDouble();
		double radius = (minDist + minDist * (d));

		d = MathUtil.random.nextDouble();
		double angle = 2 * Math.PI * (d);

		double newX = radius * Math.sin(angle);
		double newY = radius * Math.cos(angle);

		Vector2DDouble randomPoint = new Vector2DDouble(centre.x + newX, centre.y + newY);

		return randomPoint;
	}
}
