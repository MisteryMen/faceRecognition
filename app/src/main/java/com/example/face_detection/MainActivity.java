package com.example.face_detection;

import java.io.File;
import java.io.IOException;
import java.lang.System;
import android.Manifest;
import android.content.pm.ActivityInfo;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Environment;
import android.graphics.BitmapFactory;
import android.provider.MediaStore;
import android.util.Log;
import java.io.InputStream;

import org.bytedeco.javacpp.presets.opencv_core;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import android.os.Bundle;

import androidx.core.app.ActivityCompat;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.content.Intent;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.util.ArrayList;
import java.util.Arrays;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.R.string;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.view.View;
import android.view.View.OnClickListener;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.os.Bundle;

import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;


public class MainActivity extends Activity implements CvCameraViewListener2{
    private String TAG = "OpenCV_Test";
    FaceRecognizer faceRecognizer;
    private CameraBridgeViewBase cameraView;
    private Mat mRgba,mTmp,mRgb;
    private CascadeClassifier classifier;
    private int mAbsoluteFaceSize = 0;
    private boolean isFrontCamera = false;

    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("opencv_java");
    }


    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this)
    {
        @Override
        public void onManagerConnected(int status){
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG,"OpenCV loaded successfully");
                    cameraView.enableView();
                    break;
                default:
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initClassifier ();
        trainFaceRecognizer();
        initWindowSettings();

        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);

        isFrontCamera = false;
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);

       // cameraView.enableFpsMeter (); // Muestra la velocidad de fotogramas
        cameraView.setMaxFrameSize(1280,720);
        cameraView.setCvCameraViewListener(this);


    }


    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug())
        {
            Log.d(TAG,"OpenCV library not found!");
        }
        else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    };

    @Override
    public void onDestroy()
    {
        super.onDestroy();
        if(cameraView!=null)
        {
            cameraView.disableView();
        }
    };

    // Procesar el inicio de la cámara
    @Override
    public void onCameraViewStarted(int width, int height)
    {
        // TODO Auto-generated method stub
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgb  = new Mat(height, width, CvType.CV_8UC3);
        mTmp = new Mat(height, width, CvType.CV_8UC4);
    }

    // Manejar la parada de la cámara
    @Override
    public void onCameraViewStopped()
    {
        // TODO Auto-generated method stub
        mRgba.release();
        mTmp.release();
    }


    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        mRgba = DetectFace(inputFrame);
        return mRgba;
    }

    public Mat DetectFace(CvCameraViewFrame inputFrame)
    {
        Mat _mRgba = inputFrame.rgba();
        Mat _mGray = inputFrame.gray();

        if (isFrontCamera)
        {
            Core.flip(_mRgba, _mRgba, 1);
            Core.flip(_mGray, _mGray, 1);
        }
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0)
        {
            int height = _mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0)
            {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();
        if (classifier != null)
            classifier.detectMultiScale(_mGray, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);
        for (Rect faceRect : facesArray)
            Imgproc.rectangle(_mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);
            recognizeFace(_mRgba);
        return _mRgba;
    }

    // Inicializar la configuración de la ventana, incluida la pantalla completa, la pantalla horizontal y la luz constante
    private void initWindowSettings() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
    }

    private void initClassifier() {
        try {
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void trainFaceRecognizer() {

        // Prepare training data
        ArrayList<Mat> images = new ArrayList<>();
        ArrayList<Integer> labels =new ArrayList<>();



        try {
            AssetManager assetManager = this.getAssets();
            String [] peopleFolders = assetManager.list("trainingImages");
            int label = 0;
            for (int i=0; i<peopleFolders.length;i++) {
                String imagePath = "trainingImages/"+peopleFolders[i];

                Mat grayImage =new Mat();
                Mat image1 = new Mat();
                Utils.bitmapToMat(getBitmapFromAsset(imagePath),image1);
                Imgproc.cvtColor(image1, grayImage, Imgproc.COLOR_BGR2GRAY,0);

                MatOfRect faceRectangles =new MatOfRect();
                classifier.detectMultiScale(grayImage,faceRectangles,1.05,0,2,
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
                Rect [] arrayRects = faceRectangles.toArray();
                for (int j =0 ;j < arrayRects.length;j++) {
                    Mat faceImage =new Mat(grayImage, arrayRects[j]);
                    images.add(faceImage);
                    labels.add(label);
                }
                label++;
            }
        }catch (Exception e){
            Log.d("CLA",e.getMessage().toString());
        }


        // Train the face recognizer
        faceRecognizer = LBPHFaceRecognizer.create();
        MatOfInt matLabels =new MatOfInt();
        matLabels.fromList(labels);
        faceRecognizer.train(images,matLabels);
    }
    public  Bitmap getBitmapFromAsset(String filePath) {
        AssetManager assetManager = this.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }

        return bitmap;
    }

    private void recognizeFace( Mat image) {
        //Loader.load(org.bytedeco.javacpp.opencv_objdetect::class.java)
        // Convert the input image to grayscale
        Mat grayImage =new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY,0);

        // Detect faces in the image
        MatOfRect faceRectangles =new MatOfRect();
        classifier.detectMultiScale(grayImage,faceRectangles,1.05,0,2,
                new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());

        Rect [] arrayRects = faceRectangles.toArray();
            for (int j =0 ;j < arrayRects.length;j++) {

            Mat faceImage = new Mat(grayImage, arrayRects[j]);
            int [] predictedLabel = new int[1];
            double [] confidence = new double [1];
            faceRecognizer.predict(faceImage, predictedLabel, confidence);

            // Display recognized label and confidence
            int label = predictedLabel[0];
            double predictedConfidence = confidence[0];
            if(confidence[0] < 70) {
                Log.d("Recognized", getName(label));
            }

        }
    }
    public String getName(int caso){
        switch (caso){
            case 0:
                return "Raj";

            case 1:
                return "Leonard";

            case 2:
                return "Sheldon";

        }
        return null;
    }
}
