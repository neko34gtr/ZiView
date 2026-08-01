using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ZiView
{
    /// <summary>
    /// 画像ビューア上に重ねて表示するオーバーレイ型のトーンカーブコントローラー。
    /// マウスで制御点の追加・移動・削除ができ、変更のたびに256階調のLUTを再計算してCurveChangedで通知する。
    /// 画像本体（Mat）は一切保持・加工しない。適用は呼び出し側（MainWindow.Display.cs側）がLUTをCv2.LUTで都度適用する。
    /// </summary>
    public partial class ToneCurveOverlay : UserControl
    {
        private const double AreaSize = 200.0; // CurveCanvasの一辺（XAML側のWidth/Heightと合わせること）
        private const double PointHitRadius = 9.0;
        private const double MinPointGapCurveUnits = 4.0; // 隣接ポイントとの最小X間隔（連続クリックでの点重なり防止）

        // 制御点はカーブ空間（X:0-255=入力値、Y:0-255=出力値）で保持し、常にX昇順を維持する。
        // 先頭(X=0)と末尾(X=255)は端点でY方向のみ動かせる（削除不可、X固定）。初期状態は左下から右上への直線＝無調整。
        private readonly List<Point> _points = new()
        {
            new Point(0, 0),
            new Point(255, 255),
        };

        private int _dragIndex = -1;

        public event EventHandler<byte[]>? CurveChanged;

        public ToneCurveOverlay()
        {
            InitializeComponent();
            Loaded += (s, e) => RedrawAll();
        }

        /// <summary>現在のカーブ形状から256階調のLUTを生成する（lut[入力値] = 出力値）。</summary>
        public byte[] BuildLut()
        {
            var lut = new byte[256];
            for (int x = 0; x < 256; x++)
            {
                lut[x] = (byte)Math.Clamp((int)Math.Round(EvaluateCurve(x)), 0, 255);
            }
            return lut;
        }

        /// <summary>制御点を初期状態（左下から右上への直線＝無調整）へ戻す。</summary>
        public void ResetCurve()
        {
            _points.Clear();
            _points.Add(new Point(0, 0));
            _points.Add(new Point(255, 255));
            _dragIndex = -1;
            RedrawAll();
            CurveChanged?.Invoke(this, BuildLut());
        }

        /// <summary>
        /// Catmull-Rom（Y値のみをtでパラメータ化した簡易版）でxにおける出力値を評価する。
        /// X間隔が不均一でも実用上十分な滑らかさが得られる、トーンカーブエディタでよく使われる簡易手法。
        /// </summary>
        private double EvaluateCurve(double x)
        {
            int n = _points.Count;
            int i = 0;
            while (i < n - 2 && _points[i + 1].X < x) i++;

            var p1 = _points[i];
            var p2 = _points[Math.Min(i + 1, n - 1)];
            var p0 = _points[Math.Max(i - 1, 0)];
            var p3 = _points[Math.Min(i + 2, n - 1)];

            double span = p2.X - p1.X;
            double t = span <= 0.0001 ? 0 : (x - p1.X) / span;
            t = Math.Clamp(t, 0, 1);

            double t2 = t * t;
            double t3 = t2 * t;

            double y = 0.5 * (
                (2 * p1.Y) +
                (-p0.Y + p2.Y) * t +
                (2 * p0.Y - 5 * p1.Y + 4 * p2.Y - p3.Y) * t2 +
                (-p0.Y + 3 * p1.Y - 3 * p2.Y + p3.Y) * t3);

            return y;
        }

        private Point CurveToCanvas(Point cp)
            => new Point(cp.X / 255.0 * AreaSize, AreaSize - cp.Y / 255.0 * AreaSize);

        private Point CanvasToCurve(Point pt)
            => new Point(
                Math.Clamp(pt.X / AreaSize * 255.0, 0, 255),
                Math.Clamp((AreaSize - pt.Y) / AreaSize * 255.0, 0, 255));

        private void RedrawAll()
        {
            if (CurveCanvas == null) return;
            CurveCanvas.Children.Clear();

            DrawGrid();
            DrawCurveLine();
            DrawPointMarkers();
        }

        private void DrawGrid()
        {
            var gridBrush = new SolidColorBrush(Color.FromRgb(90, 90, 90));
            for (int i = 1; i <= 3; i++)
            {
                double pos = AreaSize * i / 4.0;

                CurveCanvas.Children.Add(new Line
                {
                    X1 = pos,
                    X2 = pos,
                    Y1 = 0,
                    Y2 = AreaSize,
                    Stroke = gridBrush,
                    StrokeThickness = 1
                });
                CurveCanvas.Children.Add(new Line
                {
                    X1 = 0,
                    X2 = AreaSize,
                    Y1 = pos,
                    Y2 = pos,
                    Stroke = gridBrush,
                    StrokeThickness = 1
                });
            }
        }

        private void DrawCurveLine()
        {
            var lut = BuildLut();
            var pts = new PointCollection();
            for (int x = 0; x < 256; x++)
            {
                pts.Add(CurveToCanvas(new Point(x, lut[x])));
            }

            var curveLine = new Polyline
            {
                Points = pts,
                Stroke = Brushes.Magenta,
                StrokeThickness = 2,
                IsHitTestVisible = false
            };
            CurveCanvas.Children.Add(curveLine);
        }

        private void DrawPointMarkers()
        {
            foreach (var cp in _points)
            {
                var canvasPt = CurveToCanvas(cp);
                var ellipse = new Ellipse
                {
                    Width = 8,
                    Height = 8,
                    Fill = Brushes.White,
                    Stroke = Brushes.Magenta,
                    StrokeThickness = 1,
                    IsHitTestVisible = false
                };
                Canvas.SetLeft(ellipse, canvasPt.X - 4);
                Canvas.SetTop(ellipse, canvasPt.Y - 4);
                CurveCanvas.Children.Add(ellipse);
            }
        }

        private int HitTestPoint(Point canvasPos)
        {
            for (int i = 0; i < _points.Count; i++)
            {
                var pt = CurveToCanvas(_points[i]);
                if ((pt - canvasPos).Length <= PointHitRadius) return i;
            }
            return -1;
        }

        private void CurveCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            var pos = e.GetPosition(CurveCanvas);
            int hit = HitTestPoint(pos);

            if (hit < 0 && e.ClickCount >= 2)
            {
                // 制御点の無い空き領域でのダブルクリックはカーブの初期化（リセット）とする
                ResetCurve();
                e.Handled = true;
                return;
            }

            if (hit >= 0)
            {
                _dragIndex = hit;
            }
            else
            {
                // 空き領域クリックで新規ポイントを追加する。既存点と近すぎる場合は誤操作防止のため追加しない
                var newCp = CanvasToCurve(pos);
                int insertAt = 0;
                while (insertAt < _points.Count && _points[insertAt].X < newCp.X) insertAt++;

                bool tooClose =
                    (insertAt > 0 && newCp.X - _points[insertAt - 1].X < MinPointGapCurveUnits) ||
                    (insertAt < _points.Count && _points[insertAt].X - newCp.X < MinPointGapCurveUnits);

                if (tooClose) return;

                _points.Insert(insertAt, newCp);
                _dragIndex = insertAt;
            }

            CurveCanvas.CaptureMouse();
            RedrawAll();
            CurveChanged?.Invoke(this, BuildLut());
            e.Handled = true;
        }

        private void CurveCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (_dragIndex < 0) return;

            var pos = e.GetPosition(CurveCanvas);
            var cp = CanvasToCurve(pos);
            bool isEndpoint = _dragIndex == 0 || _dragIndex == _points.Count - 1;

            double x = isEndpoint ? _points[_dragIndex].X : cp.X;
            if (!isEndpoint)
            {
                double minX = _points[_dragIndex - 1].X + MinPointGapCurveUnits;
                double maxX = _points[_dragIndex + 1].X - MinPointGapCurveUnits;
                if (maxX < minX) maxX = minX;
                x = Math.Clamp(x, minX, maxX);
            }

            _points[_dragIndex] = new Point(x, cp.Y);

            RedrawAll();
            CurveChanged?.Invoke(this, BuildLut());
            e.Handled = true;
        }

        private void CurveCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _dragIndex = -1;
            CurveCanvas.ReleaseMouseCapture();
        }

        private void CurveCanvas_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            var pos = e.GetPosition(CurveCanvas);
            int hit = HitTestPoint(pos);

            // 端点（0番目・末尾）は削除不可。カーブが未定義にならないよう最低2点を維持する
            if (hit > 0 && hit < _points.Count - 1)
            {
                _points.RemoveAt(hit);
                _dragIndex = -1;
                RedrawAll();
                CurveChanged?.Invoke(this, BuildLut());
            }
            e.Handled = true;
        }
    }
}
