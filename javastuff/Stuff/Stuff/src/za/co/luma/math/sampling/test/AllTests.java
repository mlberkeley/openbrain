package za.co.luma.math.sampling.test;

import junit.framework.Test;
import junit.framework.TestSuite;

public class AllTests
{

	public static Test suite()
	{
		TestSuite suite = new TestSuite("Test for za.co.luma.math.sampling.test");
		//$JUnit-BEGIN$
		suite.addTestSuite(TestPoissonDiskMultiSampler.class);
		//$JUnit-END$
		return suite;
	}

}
