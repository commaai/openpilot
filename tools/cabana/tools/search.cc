#include "tools/cabana/tools/search.h"

#include <iostream>


std::map<ScanType, std::string> scanTypeToDisplayName {
  {ExactValue, "Exact value"},
  {BiggerThan, "Bigger than..."},
  {SmallerThan, "Smaller than..."},
  {BiggerThan, "Bigger than..."},
  {ValueBetween, "Value between..."},
  {IncreasedValue, "Increased value"},
  {IncreaseValueBy, "Increased value by..."},
  {DecreasedValue, "Decreased value"},
  {DecreasedValueBy, "Decreased value by..."},
  {ChangedValue, "Changed value"},
  {UnchangedValue, "Unchanged value"},
  {UnknownInitialValue, "Unknown initial value"},
};

uint64_t getBitValue(uint64_t val, int offset, int size){
    return (((1 << size) - 1) & (val >> (offset - 1)));
}

SearchDlg::SearchDlg(QWidget *parent) : QDialog(parent) {
    setWindowTitle(tr("Search"));
    setAttribute(Qt::WA_DeleteOnClose);
    
    QVBoxLayout *main_layout = new QVBoxLayout(this);

    QHBoxLayout *scan_button_layout = new QHBoxLayout();

    first_scan_button = new QPushButton(QString("..."), this);
    next_scan_button = new QPushButton(QString("Next Scan"), this);
    undo_scan_button = new QPushButton(QString("Undo Scan"), this);

    undo_scan_button->setEnabled(false);
    next_scan_button->setEnabled(false);

    QObject::connect(first_scan_button, &QPushButton::clicked, [=]() { firstScan(); });
    QObject::connect(next_scan_button, &QPushButton::clicked, [=]() { nextScan(); });
    QObject::connect(undo_scan_button, &QPushButton::clicked, [=]() { undoScan(); });

    scan_button_layout->addWidget(first_scan_button);
    scan_button_layout->addWidget(next_scan_button);
    scan_button_layout->addWidget(undo_scan_button);

    QVBoxLayout *search_parameters_layout = new QVBoxLayout();

    QLineEdit *value_box1 = new QLineEdit();
    value_box1->setValidator( new QIntValidator(this) );

    QObject::connect(value_box1, &QLineEdit::textChanged, [=](QString value) { scan_value1 = value.toInt(); });

    scan_type = new QComboBox();

    QObject::connect(scan_type, qOverload<int>(&QComboBox::currentIndexChanged), [=](int index) { selectedScanType = (ScanType)(scan_type->itemData(index).toInt()); });

    QHBoxLayout *bits_min_max_layout = new QHBoxLayout();

    QSpinBox *bits_min = new QSpinBox();
    bits_min->setRange(1,32);
    bits_min->setValue(scan_bits_range_min);

    QSpinBox *bits_max = new QSpinBox();
    bits_max->setRange(1,32);
    bits_max->setValue(scan_bits_range_max);

    QObject::connect(bits_min, qOverload<int>(&QSpinBox::valueChanged), [=](int value) { scan_bits_range_min=value; });
    QObject::connect(bits_max, qOverload<int>(&QSpinBox::valueChanged), [=](int value) { scan_bits_range_max=value; });

    bits_min_max_layout->addWidget(bits_min);
    bits_min_max_layout->addWidget(bits_max);

    search_parameters_layout->addWidget(value_box1);
    search_parameters_layout->addWidget(scan_type);
    search_parameters_layout->addLayout(bits_min_max_layout);

    QVBoxLayout *search_results_layout = new QVBoxLayout();

    numberOfSigsLabel = new QLabel(QString("Found: 0"));
    search_results_layout->addWidget(numberOfSigsLabel);

    data_table = new QTableWidget();
    data_table->setRowCount(1);
    data_table->setColumnCount(6);

    search_results_layout->addWidget(data_table);

    main_layout->addLayout(scan_button_layout);
    main_layout->addLayout(search_parameters_layout);
    main_layout->addLayout(search_results_layout);

    update();
}

void SearchDlg::setRowData(int row, QString msgID, QString bitRange, QString currentValue, QString previousValue){
    QTableWidgetItem *msg_id = new QTableWidgetItem(msgID);
    data_table->setItem(row, 0, msg_id);

    QTableWidgetItem *bit_range = new QTableWidgetItem(bitRange);
    data_table->setItem(row, 1, bit_range);

    QTableWidgetItem *current_value = new QTableWidgetItem(currentValue);
    data_table->setItem(row, 2, current_value);

    QTableWidgetItem *previous_value = new QTableWidgetItem(previousValue);
    data_table->setItem(row, 3, previous_value);
}

void SearchDlg::update(){
    first_scan_button->setText(scanningStarted ? "New Scan" : "First Scan");
    numberOfSigsLabel->setText(QString("Found: ") + QString::number(filteredSignals.size()));

    ScanType selectedValue = (ScanType)(scan_type->currentData().toInt());

    next_scan_button->setEnabled(scanningStarted);
    undo_scan_button->setEnabled(false);

    scan_type->clear();

    int selectedIndex = -1;
    int i = 0;

    for(auto scanType : enabledScanTypes()){
        if(scanType == selectedValue) selectedIndex = i;
        scan_type->addItem(QString::fromStdString(scanTypeToDisplayName[scanType]), QVariant(scanType));
        i++;
    }

    if(selectedIndex != -1){
        scan_type->setCurrentIndex(selectedIndex);
    }

    data_table->clear();
    data_table->setRowCount(filteredSignals.size() + 1);

    setRowData(0, QString("Message ID"), QString("Bit Range"), QString("Current Value"), QString("Previous Value"));

    int row=1;
    for(auto &sig : filteredSignals){
        setRowData(row, sig.messageID.toString(), QString("%1:%2").arg(sig.offset).arg(sig.offset+sig.size), QString::number(sig.getValue()), QString::number(sig.previousValue));
        row++;
    }

    for(auto &sig : filteredSignals){
        sig.previousValue = sig.getValue();
    }
}

std::vector<Sig> getAllPossibleSignals(int bits_min, int bits_max){
    std::vector<Sig> ret;
    
    for(auto msg_id : can->last_msgs.keys()) {
        for(int i = bits_min; i < bits_max+1; i++) {
            for(int j = 0; j < 64 - i; j++) {
                ret.push_back(Sig(msg_id, j, i));
            }
        }
    }

    return ret;
}

std::vector<ScanType> SearchDlg::enabledScanTypes(){
    if(!scanningStarted){
        return std::vector<ScanType> {
            ExactValue,
            BiggerThan,
            SmallerThan,
            ValueBetween,
            UnknownInitialValue
        };
    }
    else{
        return std::vector<ScanType> {
            ExactValue,
            BiggerThan,
            SmallerThan,
            ValueBetween,
            IncreasedValue,
            IncreaseValueBy,
            DecreasedValue,
            DecreasedValueBy,
            ChangedValue,
            UnchangedValue
        };
    }
}

void SearchDlg::firstScan(){
    if(scanningStarted){
        filteredSignals.clear();
        scanningStarted = false;
    }
    else{
        filteredSignals = getAllPossibleSignals(scan_bits_range_min, scan_bits_range_max);
        scanningStarted = true;

        nextScan();
    }
    update();
}

SignalFilterer* SearchDlg::getCurrentFilterer() {
    if(selectedScanType == ExactValue){
        return new ExactValueSignalFilterer(scan_value1);
    }
    if(selectedScanType == BiggerThan){
        return new BiggerThanSignalFilterer(scan_value1);
    }
    if(selectedScanType == SmallerThan){
        return new SmallerThanSignalFilterer(scan_value1);
    }
    if(selectedScanType == UnknownInitialValue){
        return new UnknownInitialValueSignalFilter();
    }
    if(selectedScanType == UnchangedValue){
        return new UnchangedValueSignalFilter();
    }
    if(selectedScanType == ChangedValue){
        return new ChangedValueSignalFilter();
    }
    if(selectedScanType == IncreasedValue){
        return new IncreasedValueSignalFilter();
    }
    if(selectedScanType == DecreasedValue){
        return new DecreasedValueSignalFilter();
    }

    return nullptr;
}

void SearchDlg::nextScan(){
    auto filterer = getCurrentFilterer();

    filteredSignals = filterer->filter(filteredSignals);

    update();
}

void SearchDlg::undoScan(){

}