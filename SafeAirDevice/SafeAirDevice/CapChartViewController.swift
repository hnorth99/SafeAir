
import UIKit
import Charts
import TinyConstraints
import Foundation
import FirebaseAuth
import FirebaseDatabase
class CapChartViewController: UIViewController, ChartViewDelegate {
    let appDelegate = UIApplication.shared.delegate as! AppDelegate


    var lineChart = LineChartView()
    override func viewDidLoad() {
        super.viewDidLoad()
        lineChart.delegate = self

    }
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        lineChart.frame = CGRect(x:0, y:0, width: self.view.frame.size.width,
                                 height: self.view.frame.size.width)
        lineChart.center = view.center
        view.addSubview(lineChart)
        var test = [ChartDataEntry]()
        var entries = [Double]()
        appDelegate.ref.child("HunterApt").getData(completion: { error, snapshot in
            guard error == nil else {
                print(error!.localizedDescription)
                return;
            }
            let entries = snapshot.value as? NSDictionary
            let cap = entries?["maxcapacity"] as? Double ?? 0.0005
            var time = entries?["time"] as? [NSString] ?? [""]
            //for x in 0..<cap.count {
            //    test.append(ChartDataEntry(x: Double(x-70), y: Double(cap[x])))
            //}
            test.append(ChartDataEntry(x: Double(1), y: Double(cap)))

            self.lineChart.xAxis.drawLabelsEnabled = false
            self.lineChart.rightAxis.drawLabelsEnabled = false
            self.lineChart.legend.enabled = false
            let set = LineChartDataSet(entries: test)
            set.colors = ChartColorTemplates.material()
            let data = LineChartData(dataSet: set)
            self.lineChart.data = data
        }
        )
    }
}
