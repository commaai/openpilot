#include "tools/cabana/tools/search.h"

#include <iostream>


std::map<ScanType, std::string> scanTypeToDisplayName {
  {ExactValue, "Exact value"},
  {BiggerThan, "Bigger than..."},
  {SmallerThan, "Smaller than..."},
  {BiggerThan, "Bigger than..."},
  {ValueBetween, "Value between..."},
  {IncreasedValue, "Increased value"},
  {IncreasedValueBy, "Increased value by..."},
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

    data_table->setContextMenuPolicy(Qt::CustomContextMenu);
    QObject::connect(data_table, &QTableWidget::customContextMenuRequested, this, &SearchDlg::showDataTableContextMenu);

    search_results_layout->addWidget(data_table);

    main_layout->addLayout(scan_button_layout);
    main_layout->addLayout(search_parameters_layout);
    main_layout->addLayout(search_results_layout);

    setMinimumSize({700, 500});

    update();

    QObject::connect(can, &AbstractStream::received, this, &SearchDlg::updateRowData);
    QObject::connect(can, &AbstractStream::seekedTo, this, &SearchDlg::updateRowData);
}

void SearchDlg::showDataTableContextMenu(const QPoint &pt){
  int index = data_table->rowAt(pt.y());

  if (index >= 0) {
    QMenu menu(this);
    menu.addAction(tr("Add To Signals"), [=](){
        cabana::Signal sig;
        sig.factor = 1;
        sig.is_little_endian = filteredSignals[index].is_little_endian;
        sig.lsb = filteredSignals[index].lsb;
        sig.msb = filteredSignals[index].msb;
        sig.size = filteredSignals[index].size;
        sig.start_bit = filteredSignals[index].start_bit;
        sig.start_bit = filteredSignals[index].start_bit;
        sig.factor = filteredSignals[index].factor;
        sig.offset = filteredSignals[index].offset;
        sig.name = "NEW_SIG_FROM_SEARCH";

        dbc()->addSignal(filteredSignals[index].messageID, sig);
    });

    if (menu.exec(data_table->mapToGlobal(pt))) {
    }
  }
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

void SearchDlg::updateRowData(){
    data_table->clear();
    data_table->setRowCount(0);

    if(filteredSignals.size() < 1000){
        data_table->setRowCount(filteredSignals.size() + 1);

        setRowData(0, QString("Message ID"), QString("Bit Range"), QString("Current Value"), QString("Previous Value"));

        int row=1;
        for(auto &sig : filteredSignals){
            setRowData(row, sig.messageID.toString(), sig.toString(), QString::number(sig.getCurrentValue()), QString::number(-1));
            row++;
        }
    }
}

void SearchDlg::update(){
    first_scan_button->setText(scanningStarted() ? "New Scan" : "First Scan");
    numberOfSigsLabel->setText(QString("Found: ") + QString::number(filteredSignals.size()));

    ScanType selectedValue = (ScanType)(scan_type->currentData().toInt());

    next_scan_button->setEnabled(scanningStarted());
    undo_scan_button->setEnabled(scanningStarted());

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

    updateRowData();
}

std::vector<SearchSignal> getAllPossibleSignals(int bits_min, int bits_max){
    std::vector<SearchSignal> ret;
    
    for(auto msg_id : can->last_msgs.keys()) {
        for(int size = bits_min; size < bits_max + 1; size++) {
            for(int start_bit = 0; start_bit < 64 - size; start_bit++) {
                ret.push_back(SearchSignal(msg_id, start_bit, size, true));
                ret.push_back(SearchSignal(msg_id, start_bit, size, false));
            }
        }
    }

    return ret;
}

std::vector<ScanType> SearchDlg::enabledScanTypes(){
    if(!scanningStarted()){
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
            IncreasedValueBy,
            DecreasedValue,
            DecreasedValueBy,
            ChangedValue,
            UnchangedValue
        };
    }
}

bool SearchDlg::scanningStarted(){
    return searchHistory.size() > 0;
}

void SearchDlg::reset(){
    // Reset scan history and signals
    filteredSignals.clear();
    searchHistory.clear();
}

void SearchDlg::firstScan(){
    if(scanningStarted()){
        reset();
    }
    else{
        filteredSignals = getAllPossibleSignals(scan_bits_range_min, scan_bits_range_max);
        nextScan();
    }
    update();
}

SignalFilterer* SearchDlg::getCurrentFilterer() {
    SignalFiltererParams params {
        .ts_scan = can->currentSec(),
        .ts_prev = -1,
        .value1 = scan_value1,
        .value2 = scan_value2
    };

    if(searchHistory.size() > 0){
        params.ts_prev = searchHistory[searchHistory.size() - 1]->params.ts_scan;
    }

    if(selectedScanType == ExactValue){
        return new ExactValueSignalFilterer(params);
    }
    if(selectedScanType == BiggerThan){
        return new BiggerThanSignalFilterer(params);
    }
    if(selectedScanType == SmallerThan){
        return new SmallerThanSignalFilterer(params);
    }
    if(selectedScanType == UnknownInitialValue){
        return new UnknownInitialValueSignalFilter(params);
    }
    if(selectedScanType == UnchangedValue){
        return new UnchangedValueSignalFilter(params);
    }
    if(selectedScanType == ChangedValue){
        return new ChangedValueSignalFilter(params);
    }
    if(selectedScanType == IncreasedValue){
        return new IncreasedValueSignalFilter(params);
    }
    if(selectedScanType == DecreasedValue){
        return new DecreasedValueSignalFilter(params);
    }

    throw std::runtime_error("Unsupported scan type...");
}

void SearchDlg::nextScan(){
    searchHistory.push_back(std::shared_ptr<SignalFilterer>(getCurrentFilterer()));

    std::cout << "Scan Started!!!" << std::endl;

    filteredSignals = searchHistory[searchHistory.size() - 1]->filter(filteredSignals);

    std::cout << "Scan Finished!!!" << std::endl;

    update();
}

void SearchDlg::undoScan(){
    auto searchToUndo = searchHistory[searchHistory.size() - 1];

    // copy signals that were filtered out by this search back into the list. TODO: resort them?
    std::copy(searchToUndo->filteredSignals.begin(), searchToUndo->filteredSignals.end(), std::back_inserter(filteredSignals));

    searchHistory.pop_back();

    // we have removed all scans, reset
    if(searchHistory.size() == 0){
        reset();
    }

    update();
}