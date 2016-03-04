package sampling;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import za.co.luma.geom.Vector2DDouble;
import za.co.luma.math.function.Falloff;
import za.co.luma.math.function.PerlinFunction2D;
import za.co.luma.math.function.RealFunction2DWrapper;
import za.co.luma.math.sampling.PoissonDiskMultiSampler;
import za.co.luma.math.sampling.PoissonDiskSampler;
import za.co.luma.math.sampling.Sampler;
import za.co.luma.math.sampling.UniformPoissonDiskSampler;

// Thanks http://code-spot.co.za/2010/04/07/poisson-disk-sampling-example-code/
public class PoissonDiskDemo
{
	
	public static void main(String[] args)
	{	
		System.out.println("Generating image 1/4...");
		samplePoisson();
		
		System.out.println("Generating image 2/4...");
		multiSamplePoisson();
		
		System.out.println("Generating image 3/4...");
		samplePoissonFalloff();
		
		System.out.println("Generating image 4/4...");
		samplePoissonPerlin();	
		
		System.out.println("Done.");
		System.exit(0);
	}
	
	public static void saveImage(BufferedImage image, File file)
	{
		try
		{
			ImageIO.write(image, "png", file);
			
		}
		catch (IOException e1)
		{
			System.out.println("Could not write the image file.");
			System.out.println("Make sure you have enough disk space, and have write access to the working folder of this demo.");
		}		
	}
	
	/**
	 * Makes a new image with a uniform Poisson disk sample.
	 */
	public static void samplePoisson()
	{
		int w = 400;
		final BufferedImage image = new BufferedImage(w, w, BufferedImage.TYPE_4BYTE_ABGR);
		
		clearImage(image);
		
		Sampler<Vector2DDouble> sampler = new UniformPoissonDiskSampler(0, 0, w, w, 10);
		List<Vector2DDouble> pointList = sampler.sample();

		int i = 0;
		
		int pointCount = pointList.size();
		
		for (Vector2DDouble point : pointList)
		{			
			Color c = new Color(255, 255 - i * 255 / pointCount, i * 255 / pointCount);
			image.setRGB((int) point.x, (int) point.y, c.getRGB());
			
			i++;
		}		

		saveImage(image, new File("poisson.png"));
	}
	
	/**
	 * Makes an image that illustrates layered multi-poisson sampling.
	 */
	public static void multiSamplePoisson()
	{
		int w = 500;
		final BufferedImage image = new BufferedImage(w, w, BufferedImage.TYPE_4BYTE_ABGR);

		clearImage(image);
		
		double[] radii = {20, 8, 2};	
		double[] minRadii = {10, 4, 1};	
		double[] minDist = {62, 18, 5};

		PoissonDiskMultiSampler sampler = new PoissonDiskMultiSampler(0, 0, w, w, minDist, minRadii, radii, null, true);
		List<PoissonDiskMultiSampler.Circle>[] pointList = sampler.sample();
		
		int i = 0;
		int k = 0;
		
		Graphics g = image.getGraphics();
		
		for (List<PoissonDiskMultiSampler.Circle> list : pointList)
		{
			
			for (PoissonDiskMultiSampler.Circle point : list)
			{
				//Color c = new Color(255, 255 - i * 255 / list.size(), i * 255 / list.size());
				//image.setRGB((int) point.x, (int) point.y, c.getRGB());
				double r;
				
				if(point.x < 0)
				{
					g.setColor(Color.RED);
					point.x *= -1;
					r = point.getRadius();
				}
				else
				{
					g.setColor(Color.WHITE);
					r = point.getRadius();
				}
					
				g.drawOval((int) (point.x - r), (int) (point.y - r), (int) (2*r), (int) (2*r));			
				
				i++;
			}
			
			k++;
		}

		saveImage(image, new File("multipoisson.png"));
	}

	/**
	 * Makes an image entirely black.
	 * 
	 * @param image
	 * The image to make black
	 */
	private static void clearImage(final BufferedImage image)
	{
		for (int i = 0; i < image.getWidth(); i++)
			for (int j = 0; j < image.getHeight(); j++)
				image.setRGB(i, j, Color.BLACK.getRGB());
	}
	
	/**
	 * Makes an image that is a Poisson Disk sample, with minimum distance driven by Perlin noise.
	 */
	public static void samplePoissonPerlin()
	{
		int w = 600;
		final BufferedImage image = new BufferedImage(w, w, BufferedImage.TYPE_4BYTE_ABGR);

		clearImage(image);

		RealFunction2DWrapper realfn = new RealFunction2DWrapper(new PerlinFunction2D(w, w, 3), 0.1, 1, 0.0001, 1);
		Sampler<Vector2DDouble> sampler = new PoissonDiskSampler(0, 0, w, w, 10, realfn);
		List<Vector2DDouble> pointList = sampler.sample();

		int i = 0;

		Color c = Color.WHITE;
		
		for (Vector2DDouble point : pointList)
		{
			image.setRGB((int) point.x, (int) point.y, c.getRGB());			
			i++;
		}

		saveImage(image, new File("poissonperlin.png"));
	}
	
	/**
	 * Makes an image with a Poisson disk samples, with radius falling off away from the centre of the image.
	 */
	public static void samplePoissonFalloff()
	{
		int w = 400;
		
		final BufferedImage image = new BufferedImage(w, w, BufferedImage.TYPE_4BYTE_ABGR);

		clearImage(image);

		Falloff realfn = new Falloff(w / 2, w / 2, w / 2, 1, .5);		
		Sampler<Vector2DDouble> sampler = new PoissonDiskSampler(0, 0, w, w, 10, realfn);		
		List<Vector2DDouble> pointList = sampler.sample();

		int i = 0;

		for (Vector2DDouble point : pointList)
		{
			Color c = new Color(255, 255 - i * 255 / pointList.size(), i * 255 / pointList.size());
			image.setRGB((int) point.x, (int) point.y, c.getRGB());
			
			i++;
		}		

		saveImage(image, new File("poissonfalloff.png"));
	}
}
