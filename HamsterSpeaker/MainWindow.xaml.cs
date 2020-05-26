
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.CognitiveServices.Speech;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using static HamsterSpeaker.mouse_click;
using System.Windows.Threading;

namespace HamsterSpeaker
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public static int cor_x, cor_y;
        public static string cur_text = "";
        private  void WriteLn(string text)
        {
            tbLog.Dispatcher.BeginInvoke(new Action(() =>
            {
                tbLog.Text += text + Environment.NewLine;
            }));
            sw1.Dispatcher.BeginInvoke(new Action(() =>
            {
                sw1.ScrollToEnd();
            }));
        }

        private async void NEWs()
        {
            await RecognizeSpeechAsync();

            WriteLn("Please press any key to continue...");
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromSeconds(1);
            timer.Tick += timer_Tick;
            timer.Start();

        }
        void timer_Tick(object sender, EventArgs e)
        {
            WriteLn("1");
            //SetCursorPos(1000, 1000);
        }
        async Task RecognizeSpeechAsync()
        {
            var config =
                SpeechConfig.FromSubscription(
                    "61752b18a5c44efcb28e7ede3d03240e",
                    "eastus");

            using var recognizer = new SpeechRecognizer(config, "ru-RU");
            WriteLn("Say something...");
            var result = await recognizer.RecognizeOnceAsync();
            switch (result.Reason)
            {
                case ResultReason.RecognizedSpeech:
                    WriteLn($"We recognized: {result.Text}");
                    break;
                case ResultReason.NoMatch:
                    WriteLn($"NOMATCH: Speech could not be recognized.");
                    break;
                case ResultReason.Canceled:
                    var cancellation = CancellationDetails.FromResult(result);
                    WriteLn($"CANCELED: Reason={cancellation.Reason}");

                    if (cancellation.Reason == CancellationReason.Error)
                    {
                        WriteLn($"CANCELED: ErrorCode={cancellation.ErrorCode}");
                        WriteLn($"CANCELED: ErrorDetails={cancellation.ErrorDetails}");
                        WriteLn($"CANCELED: Did you update the subscription info?");
                    }
                    break;
            }
        }
        public static System.Drawing.Image Screenshort (){

            double a = SystemParameters.VirtualScreenWidth;
            double b = SystemParameters.VirtualScreenHeight;

            Bitmap printscreen = new Bitmap(Convert.ToInt32(a), Convert.ToInt32(b));

            Graphics graphics = Graphics.FromImage(printscreen as System.Drawing.Image);

            graphics.CopyFromScreen(0, 0, 0, 0, printscreen.Size);
            using (var m = new MemoryStream())
            {
                printscreen.Save(m, System.Drawing.Imaging.ImageFormat.Jpeg);

                var img = System.Drawing.Image.FromStream(m);

                //TEST
                img.Save("d:\\test.jpg");
                

                return img;
            }

        }
        private static VectorOfPoint ProcessImageFLANN(Image<Gray, byte> template, Image<Gray, byte> sceneImage)
        {
            try
            {
                // initialization
                VectorOfPoint finalPoints = null;
                Mat homography = null;
                VectorOfKeyPoint templateKeyPoints = new VectorOfKeyPoint();
                VectorOfKeyPoint sceneKeyPoints = new VectorOfKeyPoint();
                Mat tempalteDescriptor = new Mat();
                Mat sceneDescriptor = new Mat();

                Mat mask;
                int k = 2;
                double uniquenessthreshold = 0.80;
                VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();

                // feature detectino and description
                KAZE featureDetector = new KAZE();
                featureDetector.DetectAndCompute(template, null, templateKeyPoints, tempalteDescriptor, false);
                featureDetector.DetectAndCompute(sceneImage, null, sceneKeyPoints, sceneDescriptor, false);


                // Matching

                //KdTreeIndexParams ip = new KdTreeIndexParams();
                //var ip = new AutotunedIndexParams();
                var ip = new LinearIndexParams();
                SearchParams sp = new SearchParams();
                FlannBasedMatcher matcher = new FlannBasedMatcher(ip, sp);


                matcher.Add(tempalteDescriptor);
                matcher.KnnMatch(sceneDescriptor, matches, k);

                mask = new Mat(matches.Size, 1, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(255));

                Features2DToolbox.VoteForUniqueness(matches, uniquenessthreshold, mask);

                int count = Features2DToolbox.VoteForSizeAndOrientation(templateKeyPoints, sceneKeyPoints, matches, mask, 1.5, 20);

                if (count >= 4)
                {
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(templateKeyPoints,
                        sceneKeyPoints, matches, mask, 5);
                }

                if (homography != null)
                {
                    System.Drawing.Rectangle rect = new System.Drawing.Rectangle(System.Drawing.Point.Empty, template.Size);
                    PointF[] pts = new PointF[]
                    {
                        new PointF(rect.Left,rect.Bottom),
                        new PointF(rect.Right,rect.Bottom),
                        new PointF(rect.Right,rect.Top),
                        new PointF(rect.Left,rect.Top)
                    };

                    pts = CvInvoke.PerspectiveTransform(pts, homography);
                    System.Drawing.Point[] points = Array.ConvertAll<PointF, System.Drawing.Point>(pts, System.Drawing.Point.Round);
                    finalPoints = new VectorOfPoint(points);
                }

                return finalPoints;
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }

        }
        public MainWindow()
        {
            InitializeComponent();
            NEWs();
            Screenshort();



        }
    }
}
