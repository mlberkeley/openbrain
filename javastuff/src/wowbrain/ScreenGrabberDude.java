package wowbrain;

import java.awt.AWTException;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.awt.image.WritableRaster;

import javax.swing.GrayFilter;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangByte;
import com.ericsson.otp.erlang.OtpErlangList;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangTuple;

import za.co.luma.geom.Vector2DDouble;
import za.co.luma.math.sampling.UniformPoissonDiskSampler;

public class ScreenGrabberDude{
	public static int[] sampleImage(BufferedImage img){
		return null;
	}

	public static void sendScreenPixels(){
		Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
		BufferedImage capture = null;
		try {
			capture = new Robot().createScreenCapture(screenRect);
		} catch (AWTException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		sendPixelsToErlang(capture);
	}

	public static void sendPixelsToErlang(BufferedImage img){
		ImageFilter filter = new GrayFilter(true, 50);  
		ImageProducer producer = new FilteredImageSource(img.getSource(), filter);  
		BufferedImage mage = (BufferedImage) Toolkit.getDefaultToolkit().createImage(producer);

		// get DataBufferBytes from Raster
		WritableRaster raster = mage.getRaster();
		DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
		
		byte[] pix = data.getData();
		
		final int POINTS = 500;
		
		UniformPoissonDiskSampler sampler = new UniformPoissonDiskSampler(0,0,mage.getWidth(),mage.getHeight(), 10, POINTS);
		
		OtpErlangObject[] erlArr = new OtpErlangObject[POINTS];
		
		int count = 0;
		for(Vector2DDouble vec : sampler.sample()){
			erlArr[count] = (OtpErlangObject) new OtpErlangByte(pix[(int) (vec.x * mage.getWidth() + vec.y)]);
			count++;
		}

		OtpErlangObject[] reply = new OtpErlangObject[2];

		OtpErlangList erlList = new OtpErlangList(erlArr);

		reply[0] = new OtpErlangAtom("pixels");

		reply[1] = erlList;

		OtpErlangTuple myTuple = new OtpErlangTuple(reply);

		Robotter.myOtpMbox.send(Robotter.pixel_pid, myTuple);
	}
}
