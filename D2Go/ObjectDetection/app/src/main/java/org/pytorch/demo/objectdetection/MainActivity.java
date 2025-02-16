// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;

public class MainActivity extends AppCompatActivity implements Runnable {

static {
    if (!NativeLoader.isInitialized()) {
        NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni");
    NativeLoader.loadLibrary("torchvision_ops");
}

    private int mImageIndex = 0;
    private String[] mTestImages = {"test1.png", "test2.jpg", "test3.png"};

    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/3"));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        }
                        else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto , 1);
                        }
                        else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
              startActivity(intent);
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "d2go_model.pt");

            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }


    @Override
    public void run() {
        Bitmap resizedBitmap = mBitmap;
//                .createScaledBitmap(mBitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);
        final float[] c = {resizedBitmap.getHeight(),resizedBitmap.getWidth(),1};

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0,0,resizedBitmap.getWidth(),resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);

        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});
        final Tensor inputTensor2 = Tensor.fromBlob(c, new long[] {1, 3});


        final long startTime = SystemClock.elapsedRealtime();
        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
        //IValue[] outputTuple = outputTuple1[0].toTuple();
        //double scale = outputTuple1[1].toDouble();

        System.out.println(outputTuple.length);
        System.out.println(Arrays.toString(outputTuple[0].toTensor().getDataAsFloatArray()) +"1");
        System.out.println("==============================================");
        //System.out.println(Arrays.toString(outputTuple[1].toTensor().getDataAsLongArray())+"2");
        //System.out.println(Arrays.toString(outputTuple[2].toTensor().getDataAsFloatArray())+"3");
        System.out.println(Arrays.toString(outputTuple[3].toTensor().getDataAsFloatArray())+"4");
        System.out.println("==============================================");
        System.out.println(Arrays.toString(outputTuple[4].toTensor().getDataAsFloatArray())+"5");
        System.out.println("==============================================");
        System.out.println(Arrays.toString(outputTuple[5].toTensor().getDataAsLongArray())+"6");
        System.out.println("==============================================");
        System.out.println(resizedBitmap.getWidth());
        System.out.println(resizedBitmap.getHeight());

//        System.out.println(Arrays.toString(inputTensor.getDataAsFloatArray()));
        //int size = outputTuple.size();
        //System.out.println(size);
        //float[] b = Objects.requireNonNull(outputTuple[Integer.parseInt("scores")]).toTensor().getDataAsFloatArray();
        //for (float v : b) System.out.println(v);
//        long[] c = outputTuple[2].getDataAsLongArray();
//        for (int j=0;j<6;j++)
//            System.out.println(c[j]);
//        System.out.println(outputTuple[0].getDataAsFloatArray().length);
//        System.out.println(outputTuple[1].getDataAsFloatArray().length);
//        System.out.println(outputTuple[2].getDataAsLongArray().length);
//        System.out.println(outputTuple[3].getDataAsFloatArray().length);
//        System.out.println(size);
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("D2Go",  "inference time (ms): " + inferenceTime);

        final Map<String, IValue> map = null;
        float[] boxesData = new float[]{};
        float[] scoresData = new float[]{};
        long[] labelsData = new long[]{};
        float[] keyPointsData = new float[]{};
        final Tensor boxesTensor = outputTuple[3].toTensor();
        final Tensor scoresTensor = outputTuple[4].toTensor();
        //final Tensor labelsTensor = map.get("pred_classes").toTensor();
        //final Tensor keyPointsTensor = map.get("pred_keypoints").toTensor();

        boxesData = boxesTensor.getDataAsFloatArray();
        scoresData = scoresTensor.getDataAsFloatArray();
        //labelsData = labelsTensor.getDataAsLongArray();
        //keyPointsData = keyPointsTensor.getDataAsFloatArray();


        final int n = 17;
        float[] outputs = new float[n * PrePostProcessor.OUTPUT_COLUMN];
        int count = 0;
        for (int i = 0; i < n; i++) {
//                if (scoresData[i] < 0.5)
//                    continue;

            outputs[PrePostProcessor.OUTPUT_COLUMN * count] = boxesData[3 * i];
            outputs[PrePostProcessor.OUTPUT_COLUMN * count + 1] = boxesData[3 * i + 1];
            outputs[PrePostProcessor.OUTPUT_COLUMN * count + 2] = boxesData[3 * i];
            outputs[PrePostProcessor.OUTPUT_COLUMN * count + 3] = boxesData[3 * i + 1];
//            outputs[PrePostProcessor.OUTPUT_COLUMN * count + 4] = scoresData[i];
            outputs[PrePostProcessor.OUTPUT_COLUMN * count + 4] = 0;
            outputs[PrePostProcessor.OUTPUT_COLUMN * count + 5] = 0;
            count++;
        }

        final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);

        runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(results);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });

    }

}
