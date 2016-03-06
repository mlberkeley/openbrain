package za.co.luma.math.sampling.test;

import java.util.List;

import za.co.iocom.math.MathUtil;
import za.co.luma.geom.Vector2DDouble;
import za.co.luma.geom.Vector2DInt;
import za.co.luma.math.sampling.PoissonDiskMultiSampler;
import za.co.luma.math.sampling.PoissonDiskMultiSampler.Circle;
import junit.framework.TestCase;

public class TestPoissonDiskMultiSampler extends TestCase
{
	double DELTA = 0.0001;
	
	
	private static void pr(String message)
	{
		System.out.println(message);
	}
	
	public void testSample()
	{
		MathUtil.random.setSeed(0);
		
		double[] minDist = {30.0};
		double[] minRadii = {15.0};
		double[] maxRadii = {15.0};
		
		double[] cellSize = {minDist[0] / Math.sqrt(2)};
		double[] gridWidth = {(int) (1024 / cellSize[0]) + 1};
		double[] gridHeight = {(int) (1024 / cellSize[0]) + 1};
		
		PoissonDiskMultiSampler sampler = new PoissonDiskMultiSampler(0, 0, 1024, 1024, minDist, minRadii, maxRadii, null, false);
		
		List<Circle>[] points = sampler.sample();
		
		assertEquals(points.length, 1);
		Vector2DDouble origin = new Vector2DDouble(0, 0);
		
		
		List<Circle>[][] grid = sampler.grids.get(0);
		
		for(Circle point : points[0])
		{
			int k = 0;
			double fraction = 1;
			boolean tooClose = false;					

			Vector2DInt qIndex = PoissonDiskMultiSampler.pointDoubleToInt(point, origin, cellSize[k]);
	
//				boolean tooClose = false;
	
			for (int i = Math.max(0, qIndex.x - 2); (i < Math.min(gridWidth[k], qIndex.x + 3)) && !tooClose; i++)
			{
				for (int j = Math.max(0, qIndex.y - 2); (j < Math.min(gridHeight[k], qIndex.y + 3)) && !tooClose; j++)
				{
					for (Circle gridPoint : grid[i][j])
					{
						double distance = Vector2DDouble.distance(gridPoint, point);
						
						if (distance < minDist[k] * fraction)
						{
							assertEquals("px " + point.x + " py " + point.y + " gpx " + gridPoint.x + " gpy " + gridPoint.y, 
									point.x, gridPoint.x, DELTA);
							assertEquals(point.y, gridPoint.y, DELTA);
							assertEquals(point.getRadius(), gridPoint.getRadius(), DELTA);
						}
						
						double space = distance - point.getRadius() - gridPoint.getRadius();
						
						if(space < 0)
						{
							if (Math.abs(point.x - gridPoint.x) > DELTA || Math.abs(point.y - gridPoint.y) > DELTA)
							{
								pr("space: " + space + " r");
								fail();
							}
						}
					}
				}
			}			
		}		
	}

	public void testGenerateRandomAround()
	{		
		Vector2DDouble centre =  new Vector2DDouble(10, 20);
		
		double minDist = 20.0;
		double minRadius = 5.0;
		double maxRadius = 7.0;
		double distanceScale = 0.0;
		double angleScale = 0.0;
		double radiusScale = 0.0;
		double expectedDistance = minDist;
		double expectedRadius = minRadius;
		double expectedX = centre.x + minDist;
		double expectedY = centre.y;
		
		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);
		
		distanceScale = 1.0;
		
		expectedDistance = 2*minDist;
		expectedRadius = minRadius;
		expectedX = centre.x + 2*minDist;
		expectedY = centre.y;
		
		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);

		distanceScale = 0.5;
		angleScale = 0.0;
		radiusScale = 0.0;
		expectedDistance = 1.5 * minDist;
		expectedRadius = minRadius;
		expectedX = centre.x + 1.5 * minDist;
		expectedY = centre.y;

		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);

		
		distanceScale = 0.0;
		angleScale = 0.0;
		radiusScale = 1.0;
		expectedDistance = minDist;
		expectedRadius = maxRadius;
		expectedX = centre.x + minDist;
		expectedY = centre.y;

		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);

		distanceScale = 0.0;
		angleScale = 0.0;
		radiusScale = 0.5;
		expectedDistance = minDist;
		expectedRadius = minRadius + 0.5 * (maxRadius - minRadius);
		expectedX = centre.x + minDist;
		expectedY = centre.y;

		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);

		distanceScale = 0.0;
		angleScale = 0.5;
		radiusScale = 0.0;
		expectedDistance = minDist;
		expectedRadius = minRadius;
		expectedX = centre.x - minDist;
		expectedY = centre.y;

		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);

		distanceScale = 0.0;
		angleScale = 0.25;
		radiusScale = 0.0;
		expectedDistance = minDist;
		expectedRadius = minRadius;
		expectedX = centre.x;
		expectedY = centre.y + minDist;

		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);
		
		distanceScale = 0.0;
		angleScale = 0.75;
		radiusScale = 0.0;
		expectedDistance = minDist;
		expectedRadius = minRadius;
		expectedX = centre.x;
		expectedY = centre.y - minDist;

		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);
		
		distanceScale = 0.5;
		angleScale = 0.75;
		radiusScale = 0.0;
		expectedDistance = 1.5*minDist;
		expectedRadius = minRadius;
		expectedX = centre.x;
		expectedY = centre.y - 1.5*minDist;

		checkCaseGenerateAround(centre, minDist, minRadius, maxRadius, distanceScale, angleScale, radiusScale, 
				expectedDistance, expectedRadius, expectedX, expectedY);
	}

	private void checkCaseGenerateAround(Vector2DDouble centre, double minDist, double minRadius, double radius, double distanceScale, double angleScale, double radiusScale, double expectedDistance, double expectedRadius, double expectedX, double expectedY)
	{
		Circle point = PoissonDiskMultiSampler.generateAround(centre, minDist, minRadius, radius,distanceScale, angleScale, radiusScale);
		
		double distance = point.distance(centre);
		
		assertEquals(expectedDistance, distance, DELTA);
		assertEquals(expectedRadius, point.getRadius(), DELTA);		
		assertEquals(expectedX, point.x, DELTA);
		assertEquals(expectedY, point.y, DELTA);
	}

}
