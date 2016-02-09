package mcerl;

import java.awt.AWTException;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.event.InputEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.io.IOException;
import java.util.Arrays;

import javax.swing.GrayFilter;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangBinary;
import com.ericsson.otp.erlang.OtpErlangByte;
import com.ericsson.otp.erlang.OtpErlangDecodeException;
import com.ericsson.otp.erlang.OtpErlangExit;
import com.ericsson.otp.erlang.OtpErlangInt;
import com.ericsson.otp.erlang.OtpErlangList;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangRangeException;
import com.ericsson.otp.erlang.OtpErlangTuple;
import com.ericsson.otp.erlang.OtpMbox;
import com.ericsson.otp.erlang.OtpNode;

/**
 *
 */
public class Program {

	public static void main(String[] args) {
		String nodeName = args.length > 0 ? args[0] : "com@Maxs-MacBook-Pro-2.local";
		String cookie = "ZWCAMBALXIPKLDBDVTZW";
		
		try {
			OtpNode localNode = new OtpNode("binterface@Maxs-MacBook-Pro-2.local");
			InputManager im = new InputManager(localNode, nodeName);
			Thread input = new Thread(im);
			input.start();
			
			PixelManager pm = new PixelManager(localNode, nodeName);
			Thread pixel = new Thread(pm);
			pixel.start();
			pm.getFrame().addKeyListener(im);
		
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}