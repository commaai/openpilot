#include "tools/cabana/analysis/fields.h"

#include <algorithm>
#include <cmath>

namespace cabana {
struct FieldExtractor::Impl {
  struct Node {
    explicit Node(std::string path) : path(std::move(path)) {}
    std::string path;
    Samples *samples = nullptr;
    struct Field {
      capnp::StructSchema::Field schema;
      std::unique_ptr<Node> child;
      bool optional;
    };
    std::vector<Field> children;
    std::vector<std::unique_ptr<Node>> elements;

    void append(double number, double time, Fields &fields) {
      if (std::isfinite(number)) {
        if (!samples) samples = &fields[path];
        samples->emplace_back(time, number);
      }
    }

    template <typename T>
    void readNumbers(capnp::DynamicList::Reader list, double time, Fields &fields) {
      auto numbers = list.as<capnp::List<T>>();
      for (size_t i = 0; i < numbers.size(); ++i) elements[i]->append(numbers[i], time, fields);
    }

    void read(capnp::DynamicValue::Reader value, double time, Fields &fields) {
      double number;
      switch (value.getType()) {
        case capnp::DynamicValue::BOOL: number = value.as<bool>(); break;
        case capnp::DynamicValue::INT: number = value.as<int64_t>(); break;
        case capnp::DynamicValue::UINT: number = value.as<uint64_t>(); break;
        case capnp::DynamicValue::FLOAT: number = value.as<double>(); break;
        case capnp::DynamicValue::ENUM: number = value.as<capnp::DynamicEnum>().getRaw(); break;
        case capnp::DynamicValue::STRUCT: {
          auto node = value.as<capnp::DynamicStruct>();
          if (children.empty()) {
            for (auto field : node.getSchema().getFields()) {
              auto type = field.getType();
              if (type.isVoid() || type.isText() || type.isData() || type.isInterface() || type.isAnyPointer()) continue;
              const bool optional = type.isStruct() || type.isList() ||
                                    field.getProto().getDiscriminantValue() != capnp::schema::Field::NO_DISCRIMINANT;
              children.push_back({field, std::make_unique<Node>(path + '/' + field.getProto().getName().cStr()), optional});
            }
          }
          for (auto &field : children) {
            if (!field.optional || node.has(field.schema)) field.child->read(node.get(field.schema), time, fields);
          }
          return;
        }
        case capnp::DynamicValue::LIST: {
          auto list = value.as<capnp::DynamicList>();
          while (elements.size() < list.size()) elements.push_back(std::make_unique<Node>(path + '/' + std::to_string(elements.size())));
          switch (list.getSchema().getElementType().which()) {
            case capnp::schema::Type::FLOAT32: readNumbers<float>(list, time, fields); break;
            case capnp::schema::Type::FLOAT64: readNumbers<double>(list, time, fields); break;
            default:
              for (size_t i = 0; i < list.size(); ++i) elements[i]->read(list[i], time, fields);
              break;
          }
          return;
        }
        default: return;
      }
      append(number, time, fields);
    }
  };
  struct Root {
    std::unique_ptr<Node> node;
    Samples *mono_time, *seconds, *valid;
  };
  explicit Impl(Fields &out) : out(out) {}
  Fields &out;
  std::map<uint16_t, Root> roots;
};

FieldExtractor::FieldExtractor(Fields &destination) : impl_(std::make_unique<Impl>(destination)) {}
FieldExtractor::~FieldExtractor() = default;

void FieldExtractor::extract(cereal::Event::Reader event) {
  if (event.which() == cereal::Event::Which::CAN || event.which() == cereal::Event::Which::SENDCAN) return;
  auto node = capnp::toDynamic(event);
  KJ_IF_MAYBE(field, node.which()) {
    const uint16_t index = field->getIndex();
    auto it = impl_->roots.find(index);
    if (it == impl_->roots.end()) {
      const std::string name = field->getProto().getName().cStr();
      const std::string path = '/' + name;
      Impl::Root root{std::make_unique<Impl::Node>(path), &impl_->out[path + "/__logMonoTime"],
                      &impl_->out[path + "/__logMonoTimeSeconds"], &impl_->out[path + "/__valid"]};
      it = impl_->roots.emplace(index, std::move(root)).first;
    }
    const double time = event.getLogMonoTime() * 1e-9;
    auto &root = it->second;
    root.node->read(node.get(*field), time, impl_->out);
    root.mono_time->emplace_back(time, event.getLogMonoTime());
    root.seconds->emplace_back(time, time);
    root.valid->emplace_back(time, event.getValid());
  }
}

void prepareFieldsMerge(const FieldsSnapshot &destination, Fields &batch) {
  for (auto &[path, samples] : batch) {
    auto old = destination.find(path);
    if (old == destination.end() || samples.empty()) continue;
    Samples merged;
    merged.reserve(old->second->size() + samples.size());
    std::merge(old->second->begin(), old->second->end(), samples.begin(), samples.end(),
               std::back_inserter(merged), [](const auto &a, const auto &b) { return a.x < b.x; });
    samples.swap(merged);
  }
}
}  // namespace cabana
