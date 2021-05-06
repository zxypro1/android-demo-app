package org.pytorch.demo.objectdetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

public class ObjectDetectionActivity extends AbstractCameraXActivity<ObjectDetectionActivity.AnalysisResult> {
    private Module mModule = null;
    private ResultView mResultView;

    static class AnalysisResult {
        private final ArrayList<Result> mResults;

        public AnalysisResult(ArrayList<Result> results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setResults(result.mResults);
        mResultView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mModule == null) {
            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "d2go_model.pt");
        }
        Bitmap bitmap = imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * bitmap.getWidth() * bitmap.getHeight());
        TensorImageUtils.bitmapToFloatBuffer(bitmap, 0,0,bitmap.getWidth(),bitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);

        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {3, bitmap.getHeight(), bitmap.getWidth()});
        //final Tensor inputTensor2 = Tensor.fromBlob(c, new long[] {1, 3});


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
        System.out.println(bitmap.getWidth());
        System.out.println(bitmap.getHeight());

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
        int n = 17;
        if (scoresData.length == 0){
            n = 0;
        }
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


        float imgScaleX = (float) bitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
            float imgScaleY = (float) bitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;
            float ivScaleX = (float) mResultView.getWidth() / bitmap.getWidth();
            float ivScaleY = (float) mResultView.getHeight() / bitmap.getHeight();

            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
            return new AnalysisResult(results);
        }

}

