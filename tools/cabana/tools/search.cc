#include "tools/cabana/tools/search.h"

#include <iostream>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

SearchDlg::SearchDlg(QWidget *parent) : QDialog(parent) {
    setWindowTitle(tr("Search"));
    setAttribute(Qt::WA_DeleteOnClose);
    
    QVBoxLayout *main_layout = new QVBoxLayout(this);

    QHBoxLayout *scan_button_layout = new QHBoxLayout();

    QPushButton *first_scan_button = new QPushButton(QString("First Scan"), this);
    QPushButton *next_scan_button = new QPushButton(QString("Next Scan"), this);
    QPushButton *undo_scan_button = new QPushButton(QString("Undo Scan"), this);

    QObject::connect(first_scan_button, &QPushButton::clicked, [=]() { firstScan(); });
    QObject::connect(next_scan_button, &QPushButton::clicked, [=]() { nextScan(); });
    QObject::connect(undo_scan_button, &QPushButton::clicked, [=]() { undoScan(); });

    scan_button_layout->addWidget(first_scan_button);
    scan_button_layout->addWidget(next_scan_button);
    scan_button_layout->addWidget(undo_scan_button);

    QVBoxLayout *search_parameters_layout = new QVBoxLayout();

    QLineEdit *value_box = new QLineEdit();
    value_box->setValidator( new QIntValidator(this) );

    QObject::connect(value_box, &QLineEdit::textChanged, [=](QString value) { scan_value = value.toInt(); });

    QComboBox *scan_type = new QComboBox();
    scan_type->addItem(QString("Exact Value"));

    QHBoxLayout *bits_min_max_layout = new QHBoxLayout();

    QSpinBox *bits_min = new QSpinBox();
    bits_min->setRange(1,32);
    bits_min->setValue(scan_bits_range_min);

    QSpinBox *bits_max = new QSpinBox();
    bits_max->setRange(1,32);
    bits_max->setValue(scan_bits_range_max);

    QObject::connect(bits_min,  qOverload<int>(&QSpinBox::valueChanged), [=](int value) { scan_bits_range_min=value; });
    QObject::connect(bits_max, qOverload<int>(&QSpinBox::valueChanged), [=](int value) { scan_bits_range_max=value; });

    bits_min_max_layout->addWidget(bits_min);
    bits_min_max_layout->addWidget(bits_max);

    search_parameters_layout->addWidget(value_box);
    search_parameters_layout->addLayout(bits_min_max_layout);

    main_layout->addLayout(scan_button_layout);
    main_layout->addLayout(search_parameters_layout);
}

uint64_t getBitValue(uint64_t val, int offset, int size){
    return (((1 << size) - 1) & (val >> (offset - 1)));
}

class Sig {
    public:
        Sig(MessageId _messageID, int _offset, int _size) : messageID(_messageID), offset(_offset), size(_size) {}

        MessageId messageID;
        size_t offset;
        size_t size;
    
        uint64_t getValue(){
            auto msg = can->can_msgs[messageID];
            uint64_t* data = (uint64_t*)(msg.dat.data());
            return getBitValue(*data, offset, size);
        }
};

std::vector<Sig> getAllPossibleSignals(int bits_min, int bits_max){
    std::vector<Sig> ret;
    
    for(auto msg_id : can->can_msgs.keys()) {
        for(int i = bits_min; i < bits_max+1; i++) {
            for(int j = 0; j < 64 - i; j++) {
                ret.push_back(Sig(msg_id, j, i));
            }
        }
    }

    return ret;
}

void SearchDlg::firstScan(){
    std::cout << scan_bits_range_min << " " << scan_bits_range_max << " " << scan_value << std::endl;

    std::vector<Sig> allPossibleValues = getAllPossibleSignals(scan_bits_range_min, scan_bits_range_max);

    std::cout << allPossibleValues.size() << std::endl;

    std::vector<Sig> filteredValues;
    std::copy_if(allPossibleValues.begin(), allPossibleValues.end(), std::back_inserter(filteredValues), [=](Sig i) {
        return i.getValue() == scan_value;
    });
 
    std::cout << filteredValues.size() << std::endl;
}

void SearchDlg::nextScan(){

}

void SearchDlg::undoScan(){

}