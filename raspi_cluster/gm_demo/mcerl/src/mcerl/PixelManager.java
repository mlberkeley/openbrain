package mcerl;

import java.awt.AWTException;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.io.IOException;

import javax.swing.GrayFilter;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangInt;
import com.ericsson.otp.erlang.OtpErlangList;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangTuple;
import com.ericsson.otp.erlang.OtpMbox;
import com.ericsson.otp.erlang.OtpNode;

public class PixelManager implements Runnable{
	private OtpNode localNode;
	private String nodeName;
	private JFrame frame;

	
	
	public JFrame getFrame() {
		return frame;
	}

	public void setFrame(JFrame frame) {
		this.frame = frame;
	}

	public PixelManager(OtpNode localNode, String nodeName) {
		this.localNode = localNode;
		this.nodeName = nodeName;
		frame = new JFrame();
		frame.getContentPane().setLayout(new FlowLayout());

		frame.pack();
		frame.setVisible(true);
	}
	
	@Override
	public void run() {
		OtpMbox mbox = localNode.createMbox();
		JLabel img = new JLabel(new ImageIcon(capture()));
		frame.getContentPane().add(img);
		
		OtpErlangInt[] erInt = new OtpErlangInt[412592 / 4];

		while(true){
			
				
			BufferedImage capturedImage = capture();
	
			ImageFilter filter = new GrayFilter(true, 50);  
			ImageProducer producer = new FilteredImageSource(capturedImage.getSource(), filter);  
			Image mage = Toolkit.getDefaultToolkit().createImage(producer);  
	
			BufferedImage grayscale = toBufferedImage(mage);
	
			img.setIcon(new ImageIcon(mage));
	
			DataBuffer imageVals = grayscale.getRaster().getDataBuffer();
	
			int j = 0;
			for(int i = 1; i < imageVals.getSize(); i += 4){
				erInt[j] = new OtpErlangInt(imageVals.getElem(i));
				j++;
			}
	
			OtpErlangObject[] msg = new OtpErlangObject[2];
			msg[0] = new OtpErlangAtom("pixels");
			msg[1] = new OtpErlangList(erInt);
			OtpErlangTuple tuple = new OtpErlangTuple(msg);
	

			mbox.send("pixelListener", nodeName, tuple);

		}
	}

	
	public static BufferedImage capture(){
		try{    
			Robot robot = new Robot();
			String format = "jpg";
			String fileName = "FullScreenshot." + format;

			Rectangle screenRect = new Rectangle(50,241, 856, 482);
			BufferedImage screenFullImage = robot.createScreenCapture(screenRect);
			return screenFullImage;
		} catch (AWTException  ex) {
			System.err.println(ex);
			return null;
		}

	}

	public static BufferedImage toBufferedImage(Image img)
	{
		if (img instanceof BufferedImage)
		{
			return (BufferedImage) img;
		}

		// Create a buffered image with transparency
		BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_BYTE_GRAY);

		// Draw the image on to the buffered image
		Graphics2D bGr = bimage.createGraphics();
		bGr.drawImage(img, 0, 0, null);
		bGr.dispose();

		// Return the buffered image
		return bimage;
	}
}
