package wowbrain;

import com.ericsson.otp.erlang.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

public class ScreenGrabberDude{
	public static void sendScreenPixels(){
		while(true){
			try{
				Thread.sleep(500);

			} catch (InterruptedException e){
			}
//			System.out.println("starting to grab screen");

			Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
			BufferedImage capture = null;
			try {
				capture = new Robot().createScreenCapture(screenRect);
			} catch (AWTException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

//			System.out.println("about to send to erlang");
			sendPixelsToErlang(capture);
		}
	}

	public static BufferedImage getGrayScale(BufferedImage inputImage){
		BufferedImage img = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		Graphics g = img.getGraphics();
		g.drawImage(inputImage, 0, 0, null);
		g.dispose();
		return img;
	}

	public static void sendPixelsToErlang(BufferedImage img){
//		System.out.println("grayfilter start");
//
//		ImageFilter filter = new GrayFilter(true, 50);
//		System.out.println("grayfilter stop");
//
//
//		ImageProducer producer = new FilteredImageSource(img.getSource(), filter);
//		System.out.println("producer done");
//
//		BufferedImage mage = (BufferedImage) Toolkit.getDefaultToolkit().createImage(producer);
//
//		System.out.println("got mage");
//
//
//		// get DataBufferBytes from Raster
//		WritableRaster raster = mage.getRaster();
//		System.out.println("got raster");


		WritableRaster raster = getGrayScale(img).getRaster();


		DataBufferByte data = (DataBufferByte) raster.getDataBuffer();

//		System.out.println("databuffer done");
		
		byte[] pix = data.getData();
		
		final int POINTS = 500;

//		System.out.println("starting to sample");
		
//		UniformPoissonDiskSampler sampler = new UniformPoissonDiskSampler(0,0,img.getWidth(),img.getHeight(), 10, POINTS);
		
		OtpErlangObject[] erlArr = new OtpErlangObject[POINTS];
		
//		int count = 0;
//		for(Vector2DDouble vec : sampler.sample()){
//			erlArr[count] = (OtpErlangObject) new OtpErlangByte(pix[(int) (vec.x * img.getWidth() + vec.y)]);
//			count++;
//		}
//		TODO fix sampling
		int sample_interval = pix.length / POINTS;

		for(int i = 0; i < POINTS; i++){
			erlArr[i] = new OtpErlangByte(pix[i * sample_interval]);
		}

//		System.out.println("done sample");


		OtpErlangObject[] reply = new OtpErlangObject[2];

		OtpErlangList erlList = new OtpErlangList(erlArr);

		reply[0] = new OtpErlangAtom("pixels");

		reply[1] = erlList;

		OtpErlangTuple myTuple = new OtpErlangTuple(reply);

//		System.out.println("sending pixels to erl");
		Robotter.myOtpMbox.send(Robotter.pixel_pid, myTuple);
	}
}
